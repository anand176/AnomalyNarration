import os
import numpy as np
import pandas as pd
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load the datasets
anomaly1_df = pd.read_csv("AnandPhone.txt")

# Combine the anomaly datasets (if there are more datasets, add them to the list)
anomaly_df = pd.concat([anomaly1_df])

# Prepare the data
X = []
no_of_timesteps = 20

# Process the combined Anomaly dataset
datasets = anomaly_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i - no_of_timesteps:i, :])

# Convert to numpy arrays
X = np.array(X)
print(X.shape)

# Reshape for Conv2D
X = X.reshape((X.shape[0], no_of_timesteps, X.shape[2], 1))
print(X.shape)

# Split the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2)

# Build the convolutional autoencoder model
input_shape = (no_of_timesteps, X_train.shape[2], 1)
x = Input(shape=input_shape)

# Encoder







conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
h = MaxPooling2D((2, 2), padding='same')(conv1_2)

# Decoder
conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)

autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adam', loss='mse')

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)




# Train the model
history = autoencoder.fit(X_train, X_train, epochs=100, batch_size=32, validation_data=(X_test, X_test),
                          callbacks=[early_stopping])

# Calculate reconstruction error for training set
train_predictions = autoencoder.predict(X_train)
train_loss = np.mean(np.power(X_train - train_predictions, 2), axis=(1, 2, 3))

# Calculate the threshold
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

# Save the model
autoencoder.save("autoencoder_demo7.h5")
print("Model saved successfully.")
