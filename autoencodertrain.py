import cv2
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import train_test_split

# Function to extract frames from a video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))  # Resize frame to a smaller size for easier training
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Path to your video file
# video_path = r'C:\Users\anand_ome3aek\Downloads\anomalynarrate\model\WIN_20240808_11_23_47_Pro.mp4'
video_path = r"C:\Users\anand_ome3aek\Downloads\anomalynarrate\model\WIN_20240808_11_43_32_Pro.mp4"


# Extract frames from the video
frames = extract_frames(video_path)
frames = frames.astype('float32') / 255.0  # Normalize pixel values

# Prepare data for the autoencoder
frames = np.expand_dims(frames, axis=-1)  # Add channel dimension

# Split the data into training and testing sets
X_train, X_test = train_test_split(frames, test_size=0.2)

# Build the autoencoder model
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

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Save the model
autoencoder.save("autoencoder_video1.h5")
print("Model saved successfully.")
