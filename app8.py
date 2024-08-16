import os
import cv2
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
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
genai.configure(api_key=('AIzaSyCRU31GS3v7eiqXLPR4gAKRigbIB2i_L4E'))

# Load autoencoder model
model = tf.keras.models.load_model("autoencoder_video_complex.h5")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to clear the processed frames folder
def clear_processed_frames_folder():
    for filename in os.listdir(app.config['PROCESSED_FRAMES_FOLDER']):
        file_path = os.path.join(app.config['PROCESSED_FRAMES_FOLDER'], filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))  # Updated frame size
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = frame.astype('float32') / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Function to detect anomalies
def detect_anomaly(autoencoder, frame):
    reconstructed = autoencoder.predict(frame)
    mse = np.mean(np.power(frame - reconstructed, 2))
    threshold = 0.0235 # Adjusted threshold based on new model
    return mse > threshold

# Function to process the video and detect anomalies
def process_video(video_path, output_dir):
    clear_processed_frames_folder()
    
    cap = cv2.VideoCapture(video_path)
    
    i = 0
    warm_up_frames = 60
    anomaly_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Detect anomaly in the frame
        if i > warm_up_frames:
            if detect_anomaly(model, preprocessed_frame):
                output_image_path = os.path.join(output_dir, f"anomalous_frame_{anomaly_count}.jpg")
                cv2.imwrite(output_image_path, frame)
                anomaly_count += 1
        i += 1

    cap.release()
    cv2.destroyAllWindows()

# Function to get Gemini AI response
def get_gemini_response_vision(input_text, image_parts):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, image_parts[0]])
    return response.text

# Dictionary to store prompt responses
prompt_responses = {}

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
    return render_template('upload2.html')

# Route to serve processed frames
@app.route('/processed_frames/<filename>')
def processed_frame(filename):
    return send_from_directory(app.config['PROCESSED_FRAMES_FOLDER'], filename)

# Route to display anomalous frames
@app.route('/anomalous_frames')
def show_anomalous_frames():
    frames = os.listdir(app.config['PROCESSED_FRAMES_FOLDER'])
    if not frames:
        return render_template('frames211.html', frames=None)
    return render_template('frames211.html', frames=frames)

# Route to show the image with a prompt form
@app.route('/show_image/<filename>', methods=['GET'])
def show_image(filename):
    return render_template('description1.html', filename=filename)

# Route to process the prompt and show the response
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
    response = get_gemini_response_vision('Give precise description of the interesting event in the scene', image_parts)

    # Store the prompt and response for later retrieval
    prompt_responses[filename] = {
        'prompt': 'Give precise description of the interesting event in the scene',
        'response': response
    }
    return redirect(url_for('show_prompt_response', filename=filename))

# Route to show the prompt response
@app.route('/prompt_response/<filename>')
def show_prompt_response(filename):
    prompt_response = prompt_responses.get(filename, {'prompt': 'Give precise description of the interesting event in the scene', 'response': 'No response available'})
    return render_template('prompt_response1.html', filename=filename, prompt=prompt_response['prompt'], response=prompt_response['response'])

if __name__ == '__main__':
    app.run(debug=True)
# def get_gemini_response_vision(input_text, image_parts):
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     response = model.generate_content([input_text, image_parts[0]])
#     return response.text

# # Dictionary to store prompt responses
# prompt_responses = {}

# # Route to upload video
# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(video_path)
#             output_dir = app.config['PROCESSED_FRAMES_FOLDER']
#             process_video(video_path, output_dir)
#             return redirect(url_for('show_anomalous_frames'))
#     return render_template('upload.html')

# # Route to serve processed frames
# @app.route('/processed_frames/<filename>')
# def processed_frame(filename):
#     return send_from_directory(app.config['PROCESSED_FRAMES_FOLDER'], filename)

# # Route to display anomalous frames
# @app.route('/anomalous_frames')
# def show_anomalous_frames():
#     frames = os.listdir(app.config['PROCESSED_FRAMES_FOLDER'])
#     if not frames:
#         return render_template('frames211.html', frames=None)
#     return render_template('frames211.html', frames=frames)

# # Route to show the image with a prompt form
# @app.route('/show_image/<filename>', methods=['GET'])
# def show_image(filename):
#     return render_template('description1.html', filename=filename)

# # Route to process the prompt and show the response
# @app.route('/process_prompt/<filename>', methods=['POST'])
# def process_prompt(filename):
#     file_path = os.path.join(app.config['PROCESSED_FRAMES_FOLDER'], filename)
#     with open(file_path, 'rb') as img_file:
#         img_bytes = img_file.read()
#     image_parts = [
#         {
#             "mime_type": "image/jpeg",
#             "data": img_bytes
#         }
#     ]
#     response = get_gemini_response_vision('Tell me whether the man is holding a phone. Reply in full sentence, do not mention anything else', image_parts)

#     # Store the prompt and response for later retrieval
#     prompt_responses[filename] = {
#         'prompt': 'Tell me whether the man is holding a phone. Reply in full sentence, do not mention anything else',
#         'response': response
#     }
#     return redirect(url_for('show_prompt_response', filename=filename))

# # Route to show the prompt response
# @app.route('/prompt_response/<filename>')
# def show_prompt_response(filename):
#     prompt_response = prompt_responses.get(filename, {'prompt': 'Tell me whether the man is holding a phone. Reply in full sentence, do not mention anything else', 'response': 'No response available'})
#     return render_template('prompt_response.html', filename=filename, prompt=prompt_response['prompt'], response=prompt_response['response'])

# if __name__ == '__main__':
#     app.run(debug=True)
