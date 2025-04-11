import numpy as np
import time
from model_baseline import NeuralNetwork

# Set the random seed for reproducibility
np.random.seed(42)

from data_loader import load_data

(X_train, y_train), (X_test, y_test) = load_data()

# Create an instance of your network
nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)

# Train the network and time the training process
start_time = time.time()
nn.train(X_train, y_train, learning_rate=0.001, epochs=40)
end_time = time.time()

# Print the training time
print(f"Training time: {end_time - start_time:.2f} seconds")

# Evaluate the model on the test set
accuracy = nn.evaluate(X_test, y_test)

print(f"Test accuracy: {accuracy:.2f}%")

# Plot the loss over epochs
nn.plot_loss()