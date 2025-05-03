# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# 1. Load the MNIST dataset
# Possible problem: Dataset download may fail due to network issues.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize the data (0-255 pixel values to 0-1 range)
# Problem: Skipping this step may slow down training or reduce accuracy.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Flatten the images (28x28 -> 784)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# 4. Build the neural network model
# Problem: Too few neurons can underfit; too many can overfit or use too much memory.
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),   # Input + hidden layer
    layers.Dropout(0.2),                                        # Prevent overfitting
    layers.Dense(10, activation='softmax')                     # Output layer (10 classes)
])

# 5. Compile the model
# Problem: Choosing wrong loss or optimizer can lead to poor training.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train the model
# Problem: Small epochs = underfitting; large = overfitting without validation
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 7. Evaluate the model
# Problem: Poor accuracy may need better tuning, more layers, or more training.
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

