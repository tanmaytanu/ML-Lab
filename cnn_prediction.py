# -----------------------------------------------
# Suppress TensorFlow debugging messages
# -----------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# -----------------------------------------------
# Import necessary libraries
# -----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Conv1D, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------
# Load the dataset
# -----------------------------------------------
df = pd.read_csv('Electric_Production.csv')  # Load the CSV file
data = df['IPG2211A2N'].values               # Extract the target time series column as a NumPy array

# -----------------------------------------------
# Scale the data to 0-1 range
# -----------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))  # Reshape for scaler and scale it

# -----------------------------------------------
# Create dataset: input (X) and target (y) using look-back window
# -----------------------------------------------
def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])        # last 'look_back' values as features
        y.append(data[i + look_back, 0])            # the next value as label
    return np.array(X), np.array(y)

look_back = 10  
X, y = create_dataset(data_scaled, look_back)

# -----------------------------------------------
# Reshape X to 3D shape for Conv1D input: (samples, timesteps, features)
# -----------------------------------------------
X = X.reshape(X.shape[0], X.shape[1], 1)

# -----------------------------------------------
# Split dataset into training and testing sets
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -----------------------------------------------
# Build the Conv1D model
# -----------------------------------------------
model = Sequential([
    Input(shape=(look_back, 1)),                   # Input layer with time steps and features
    Conv1D(64, kernel_size=3, activation='relu'),  # 1D convolution layer to capture temporal features
    Flatten(),                                     # Flatten the output to feed into Dense layers
    Dense(32, activation='relu'),                  # Fully connected hidden layer
    Dense(1)                                       # Output layer for regression
])

# -----------------------------------------------
# Compile the model with optimizer and loss
# -----------------------------------------------
model.compile(optimizer='adam', loss='mean_squared_error')

# -----------------------------------------------
# Train the model on training data
# -----------------------------------------------
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# -----------------------------------------------
# Predict on test set
# -----------------------------------------------
predicted = model.predict(X_test)

# -----------------------------------------------
# Inverse transform predictions and actual values to original scale
# -----------------------------------------------
predicted_rescaled = scaler.inverse_transform(predicted)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# -----------------------------------------------
# Plot actual vs predicted results
# -----------------------------------------------
plt.plot(y_test_rescaled, color='blue', label='Actual')        # Actual values
plt.plot(predicted_rescaled, color='red', label='Predicted')   # Predicted values
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

