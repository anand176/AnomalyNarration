import os
import cv2
import threading
import numpy as np
import tensorflow as tf
import h5py
import json
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify, Response, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential
from keras.initializers import Orthogonal
import mediapipe as mp
import google.generativeai as genai
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FRAMES_FOLDER'] = 'processed_frames'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FRAMES_FOLDER'], exist_ok=True)

# Load environment variables
load_dotenv()
genai.configure(api_key='apikey')

# Register custom classes and initializers
@tf.keras.utils.register_keras_serializable(package="Custom", name="Sequential")
class CustomSequential(Sequential):
    pass

@tf.keras.utils.register_keras_serializable(package="Custom", name="Orthogonal")
class CustomOrthogonal(Orthogonal):
    pass

# Load model
custom_objects = {
    'Sequential': CustomSequential,
    'Orthogonal': CustomOrthogonal
}

with h5py.File("demo3.h5", 'r') as f:
    model_config = f.attrs.get('model_config')
    model_config = json.loads(model_config)  
    for layer in model_config['config']['layers']:
        if 'time_major' in layer['config']:
            del layer['config']['time_major']
    model_json = json.dumps(model_config)
    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
    weights_group = f['model_weights']
    for layer in model.layers:
        layer_name = layer.name
        if layer_name in weights_group:
            weight_names = weights_group[layer_name].attrs['weight_names']
            layer_weights = [weights_group[layer_name][weight_name] for weight_name in weight_names]
            layer.set_weights(layer_weights)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to clear the processed frames folder
def clear_processed_frames_folder():
    for filename in os.listdir(app.config['PROCESSED_FRAMES_FOLDER']):
        file_path = os.path.join(app.config['PROCESSED_FRAMES_FOLDER'], filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

# Function to process the video and detect anomalies
def process_video(video_path, output_dir):
    # Clear the processed frames folder
    clear_processed_frames_folder()
    
    cap = cv2.VideoCapture(video_path)
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    lm_list = []
    label = "NotAnomaly"
    neutral_label = "Anomaly"

    def make_landmark_timestep(results):
        c_lm = []
        for lm in results.pose_landmarks.landmark:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm

    def draw_landmark_on_image(mpDraw, results, frame):
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for lm in results.pose_landmarks.landmark:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        return frame

    def draw_class_on_image(label, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
        thickness = 2
        lineType = 2
        cv2.putText(img, str(label),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        return img

    def detect(model, lm_list):
        nonlocal label
        lm_list = np.array(lm_list)
        lm_list = np.expand_dims(lm_list, axis=0)
        result = model.predict(lm_list)
        predicted_class = np.argmax(result[0])
        label = "Anomaly" if predicted_class == 0 else "NotAnomaly"
        return str(label)

    i = 0
    warm_up_frames = 60
    anomaly_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        i += 1
        if i > warm_up_frames:
            if results.pose_landmarks:
                lm = make_landmark_timestep(results)
                lm_list.append(lm)
                if len(lm_list) == 20:
                    t1 = threading.Thread(target=detect, args=(model, lm_list))
                    t1.start()
                    t1.join()  # Ensure the thread completes before proceeding
                    if label == "Anomaly":
                        output_image_path = os.path.join(output_dir, f"anomalous_frame_{anomaly_count}.jpg")
                        cv2.imwrite(output_image_path, frame)
                        anomaly_count += 1
                    lm_list = []
                x_coordinate = []
                y_coordinate = []
                for lm in results.pose_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_coordinate.append(cx)
                    y_coordinate.append(cy)
                cv2.rectangle(frame,
                              (min(x_coordinate), max(y_coordinate)),
                              (max(x_coordinate), min(y_coordinate) - 25),
                              (0, 255, 0),
                              1)
                frame = draw_landmark_on_image(mpDraw, results, frame)
            frame = draw_class_on_image(label, frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to handle live video streaming with anomaly detection
def generate_live_frames():
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    lm_list = []
    label = "NotAnomaly"
    neutral_label = "Anomaly"
    anomaly_count = 0

    def make_landmark_timestep(results):
        c_lm = []
        for lm in results.pose_landmarks.landmark:
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm

    def draw_landmark_on_image(mpDraw, results, frame):
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for lm in results.pose_landmarks.landmark:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        return frame

    def draw_class_on_image(label, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
        thickness = 2
        lineType = 2
        cv2.putText(img, str(label),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        return img

    def detect(model, lm_list):
        nonlocal label
        lm_list = np.array(lm_list)
        lm_list = np.expand_dims(lm_list, axis=0)
        result = model.predict(lm_list)
        predicted_class = np.argmax(result[0])
        label = "Anomaly" if predicted_class == 0 else "NotAnomaly"
        return str(label)

    i = 0
    warm_up_frames = 60

    while live_streaming:
        success, frame = cap.read()
        if not success:
            break
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        i += 1
        if i > warm_up_frames:
            if results.pose_landmarks:
                lm = make_landmark_timestep(results)
                lm_list.append(lm)
                if len(lm_list) == 20:
                    t1 = threading.Thread(target=detect, args=(model, lm_list))
                    t1.start()
                    t1.join()  # Ensure the thread completes before proceeding
                    if label == "Anomaly":
                        output_image_path = os.path.join(app.config['PROCESSED_FRAMES_FOLDER'], f"anomalous_frame_{anomaly_count}.jpg")
                        cv2.imwrite(output_image_path, frame)
                        anomaly_count += 1
                    lm_list = []
                frame = draw_landmark_on_image(mpDraw, results, frame)
            frame = draw_class_on_image(label, frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# Route to upload video
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            output_dir = app.config['PROCESSED_FRAMES_FOLDER']
            process_video(video_path, output_dir)
            return redirect(url_for('show_anomalous_frames'))
    return render_template('upload1.html')

# Route to handle live video streaming
@app.route('/live_feed')
def live_feed():
    global live_streaming
    live_streaming = True
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to start live streaming
@app.route('/start_live')
def start_live():
    return redirect(url_for('live_feed'))

# Route to stop live streaming
@app.route('/stop_live')
def stop_live():
    global live_streaming
    live_streaming = False
    return redirect(url_for('show_anomalous_frames'))

# Route to serve processed frames
@app.route('/processed_frames/<filename>')
def processed_frame(filename):
    return send_from_directory(app.config['PROCESSED_FRAMES_FOLDER'], filename)

# Route to display anomalous frames
@app.route('/anomalous_frames')
def show_anomalous_frames():
    frames = os.listdir(app.config['PROCESSED_FRAMES_FOLDER'])
    if not frames:
        return render_template('frames21.html', frames=None)
    return render_template('frames21.html', frames=frames)

# Route to show the image with a prompt form
@app.route('/show_image/<filename>', methods=['GET'])
def show_image(filename):
    return render_template('description1.html', filename=filename)


# Route to process the prompt and show the response
# @app.route('/process_prompt/<filename>', methods=['POST'])
# def process_prompt(filename):
#     prompt = request.form['prompt']
#     file_path = os.path.join(app.config['PROCESSED_FRAMES_FOLDER'], filename)
#     with open(file_path, 'rb') as img_file:
#         img_bytes = img_file.read()
#     image_parts = [
#         {
#             "mime_type": "image/jpeg",
#             "data": img_bytes
#         }
#     ]
#     response = get_gemini_response_vision(prompt, image_parts)

#     # Store the prompt and response for later retrieval
#     prompt_responses[filename] = {
#         'prompt': prompt,
#         'response': response
#     }
#     return redirect(url_for('show_prompt_response', filename=filename))

@app.route('/process_prompt/<filename>', methods=['POST'])
def process_prompt(filename):
    file_path = os.path.join(app.config['PROCESSED_FRAMES_FOLDER'], filename)
    with open(file_path, 'rb') as img_file:
        img_bytes = img_file.read()
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": img_bytes
        }
    ]
    response = get_gemini_response_vision('tell me whether the man is holding a phone. Reply in full sentence, do not mention anything else', image_parts)

    # Store the prompt and response for later retrieval
    prompt_responses[filename] = {
        'prompt': 'Describe this image',
        'response': response
    }
    return redirect(url_for('show_prompt_response', filename=filename))


# Route to show the prompt and its response
# @app.route('/prompt_response/<filename>')
# def show_prompt_response(filename):
#     prompt_response = prompt_responses.get(filename, {'prompt': 'N/A', 'response': 'N/A'})
#     return render_template('prompt_response.html', filename=filename, prompt=prompt_response['prompt'], response=prompt_response['response'])

@app.route('/prompt_response/<filename>')
def show_prompt_response(filename):
    prompt_response = prompt_responses.get(filename, {'prompt': 'tell me whether the man is holding a phone. Reply in full sentence, do not mention anything else', 'response': 'No response available'})
    return render_template('prompt_response1.html', filename=filename, prompt=prompt_response['prompt'], response=prompt_response['response'])


# def get_gemini_response_vision(input_text, image_parts, prompt=None):
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     if prompt:
#         response = model.generate_content([input_text, image_parts[0], prompt])
#     else:
#         response = model.generate_content([input_text, image_parts[0]])
#     return response.text
def get_gemini_response_vision(input_text, image_parts):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, image_parts[0]])
    return response.text


# Dictionary to store prompt responses
prompt_responses = {}

# Route to render the live video feed page
@app.route('/live')
def live():
    return render_template('live_feed2.html')

if __name__ == '__main__':
    live_streaming = False
    app.run(debug=True)
