from numba_model import NeuralNetwork
from numba import config #type: ignore[import]
print("NUMBA_DEFAULT_NUM_THREADS:", config.NUMBA_DEFAULT_NUM_THREADS)

# Set the number of threads to 10
config.NUMBA_DEFAULT_NUM_THREADS = 10

#load data
import numpy as np
from data_loader import load_data

# Set seed
np.random.seed(42)

(X_train, y_train), (X_test, y_test) = load_data()

#initialize model
nn = NeuralNetwork(784, 128, 10)

# Warm up
print("Warming up...")
nn.warmup()

# Train the model and time the training process
import time
start_time = time.time()
nn.train(X_train, y_train, learning_rate=0.001, epochs=40)
end_time = time.time()

# Print the training time
print(f"Training time: {end_time - start_time:.2f} seconds")

# Evaluate the model
accuracy = nn.evaluate(X_test, y_test)

print(f"Test accuracy: {accuracy:.2f}%")

# Plot loss
nn.plot_loss()
