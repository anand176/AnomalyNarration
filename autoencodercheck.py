import cv2
import numpy as np
import tensorflow as tf
import threading
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from sklearn.model_selection import train_test_split

# Function to build the autoencoder model
def build_autoencoder():
    input_img = Input(shape=(64, 64, 1))  # Input shape for grayscale images

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

# Load the pre-trained autoencoder model
autoencoder = build_autoencoder()
autoencoder.load_weights("autoencoder_video1.h5")

# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = frame.astype('float32') / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Function to detect anomalies
def detect_anomaly(autoencoder, frame):
    reconstructed = autoencoder.predict(frame)
    mse = np.mean(np.power(frame - reconstructed, 2))
    threshold = 0.004 # Adjust threshold based on your needs
    print(mse)
    if mse > threshold:
        return "Anomaly"
    else:
        return "NotAnomaly"

cap = cv2.VideoCapture(0)
label = "NotAnomaly"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame for the autoencoder
    preprocessed_frame = preprocess_frame(frame)

    # Detect anomaly in the frame
    label = detect_anomaly(autoencoder, preprocessed_frame)

    # Draw the label on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0) if label == "NotAnomaly" else (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(frame, str(label), bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
