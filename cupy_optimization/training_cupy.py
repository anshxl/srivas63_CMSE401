from gpu_model import GPUNeuralNetwork
import cupy as cp # type: ignore[import]
import numpy as np
import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import load_data

# Load data
(X_train, y_train), (X_test, y_test) = load_data()

# Convert data to CuPy arrays
X_train_GPU = cp.asarray(X_train)
y_train_GPU = cp.asarray(y_train)
X_test_GPU = cp.asarray(X_test)
y_test_GPU = cp.asarray(y_test)

# Initialize model
input_size = X_train_GPU.shape[1]
hidden_size = 128  # example hidden layer size
output_size = y_train_GPU.shape[1]

nn = GPUNeuralNetwork(input_size, hidden_size, output_size)

# Train the model and time the training process
import time
start_time = time.time()
nn.train(X_train_GPU, y_train_GPU, learning_rate=0.001, epochs=40)
end_time = time.time()

# Print the training time
print(f"Training time: {end_time - start_time:.2f} seconds")

# Evaluate the model
accuracy = nn.evaluate(X_test_GPU, y_test_GPU)
print(f"Test accuracy: {accuracy:.2f}%")

# Save loss plots
nn.plot_loss(show_inline=False)


