import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------------------
# 1. Generate synthetic time series data
# -------------------------------
def generate_time_series_data(samples=1000, timesteps=50):
    X = np.random.randn(samples, timesteps)
    y = np.random.randn(samples)
    return X, y

# Generate data
X, y = generate_time_series_data()

# -------------------------------
# 2. Preprocess the data using MinMaxScaler
# -------------------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Flatten to 2D for scaling
X_scaled = scaler_X.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# -------------------------------
# 3. Reshape 2D to 3D for LSTM
# LSTM expects input shape: (samples, timesteps, features)
# -------------------------------
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# -------------------------------
# Split into training and testing sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# -------------------------------
# 4. Build a simple LSTM model for univariate prediction
# -------------------------------
model = Sequential()

# First LSTM layer
model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Prevent overfitting

# Second LSTM layer (stacked)
model.add(LSTM(units=32, activation='tanh', return_sequences=False))
model.add(Dropout(0.2))  # Prevent overfitting

# Output layer for regression
model.add(Dense(1, activation='linear'))

# -------------------------------
# Compile the model
# -------------------------------
model.compile(optimizer='adam', loss='mse')

# -------------------------------
# Train the model
# -------------------------------
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32, verbose=1)

# -------------------------------
# Evaluate the model
# -------------------------------
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest Loss: {loss:.4f}')

# -------------------------------
# Predict and visualize
# -------------------------------
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_test_rescaled = scaler_y.inverse_transform(y_test)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.title('LSTM Prediction vs Actual (Univariate Time Series)')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()

