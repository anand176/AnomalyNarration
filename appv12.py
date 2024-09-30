import cv2
import numpy as np
import os
import time
import google.generativeai as genai
import streamlit as st
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from dotenv import load_dotenv
from moviepy.editor import ImageSequenceClip, VideoFileClip

# Set up media folder
MEDIA_FOLDER = 'medias'
os.makedirs(MEDIA_FOLDER, exist_ok=True)

# Load environment variables
load_dotenv()

# Set the Gemini API key
api_key = ''
genai.configure(api_key=api_key)

# Function to build the autoencoder model
def build_autoencoder():
    input_img = Input(shape=(128, 128, 1))
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return autoencoder

# Load the pre-trained autoencoder model
autoencoder = build_autoencoder()
autoencoder.load_weights("autoencoder_video_complex.h5")

# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=-1)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to detect anomalies
def detect_anomaly(autoencoder, frame):
    reconstructed = autoencoder.predict(frame)
    mse = np.mean(np.power(frame - reconstructed, 2))
    threshold = 0.0235
    return mse > threshold

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(MEDIA_FOLDER, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())
    return file_path

def extract_anomalous_frames(video_path, max_anomalous_frames=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    anomalous_frames = []
    buffer_frames = []
    found_first_anomaly = False

    while True:
        ret, frame = cap.read()
        if not ret or len(anomalous_frames) >= max_anomalous_frames:
            break

        preprocessed_frame = preprocess_frame(frame)
        
        if not found_first_anomaly:
            # Keep track of the last 50 frames in case we find an anomaly later
            buffer_frames.append(frame)
            if len(buffer_frames) > 50:
                buffer_frames.pop(0)

        if detect_anomaly(autoencoder, preprocessed_frame):
            if not found_first_anomaly:
                # Add the 50 frames before the first anomaly
                anomalous_frames.extend(buffer_frames)
                found_first_anomaly = True

            anomalous_frames.append(frame)

        frame_count += 1

    cap.release()
    return anomalous_frames

def create_anomalous_video(frames, output_path):
    if not frames:
        return

    images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    clip = ImageSequenceClip(images, fps=20)
    clip.write_videofile(output_path, codec='libx264')

def get_insights(video_path):
    st.write(f"Processing video: {video_path}")

    st.write(f"Uploading file...")
    video_file = genai.upload_file(path=video_path)
    st.write(f"Completed upload: {video_file.uri}")

    while video_file.state.name == "PROCESSING":
        st.write('Waiting for video to be processed.')
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)

    prompt = "Describe the anomaly in the scene in a single sentence"
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    st.write("Making LLM inference request...")
    response = model.generate_content([prompt, video_file], request_options={"timeout": 600})
    st.write(f'Video processing complete')
    st.subheader("Insights")
    st.write(response.text)
    genai.delete_file(video_file.name)

def app():
    st.markdown("""
        <style>
            .main {
                background-color: #e0f7fa;
                color: #00695c;
                font-family: 'Arial', sans-serif;
            }
            .stButton>button {
                background-color: #00695c;
                color: white;
                border-radius: 8px;
                font-size: 18px;
            }
            .stVideo {
                border: 2px solid #00695c;
                border-radius: 10px;
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                color: #004d40;
            }
            hr {
                border: 2px solid #004d40;
                border-radius: 5px;
            }
            .stMarkdown p {
                color: #00796b;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Anomalous Video Detection and Insights Generator")
    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.markdown("### Uploaded Video")
        st.video(file_path, format="video/mp4", start_time=0)

        st.write("Processing the video to detect anomalies...")
        anomalous_frames = extract_anomalous_frames(file_path)
        
        if anomalous_frames:
            anomalous_video_path = os.path.join(MEDIA_FOLDER, "anomalous_video.mp4")
            create_anomalous_video(anomalous_frames, anomalous_video_path)

            st.markdown("### Anomalous Video")
            st.video(anomalous_video_path, format="video/mp4", start_time=0)

            st.write("Generating insights for the anomalous video...")
            get_insights(anomalous_video_path)
        else:
            st.write("No anomalies detected.")
    
    st.markdown("<hr>", unsafe_allow_html=True)

if __name__ == "__main__":
    load_dotenv()
    app()
