from typing import Sequence
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Load the datasets
anomaly1_df = pd.read_csv("Anomaly.txt")
# anomaly2_df = pd.read_csv("Anomaly2.txt")
# anomaly3_df = pd.read_csv("Anomaly3.txt")
# anomaly4_df = pd.read_csv("Anomaly4.txt")

# Combine the anomaly datasets
anomaly_df = pd.concat([anomaly1_df])

# Prepare the data
X = []
y = []
no_of_timesteps = 20

# Process the combined Anomaly dataset
datasets = anomaly_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(0) # Label for Anomaly

# Convert to numpy arrays
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation="sigmoid"))  # 1 unit for binary classification

# Compile the model
model.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy")

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save("demo1.h5")
