import numpy as np
from numba import njit #type: ignore
import sys
import os
import time
from tqdm import tqdm  # type: ignore

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adam import AdamOptimizer

# Standalone functions decorated with fastmath
@njit(fastmath=True)
def relu_core(x):
    return np.maximum(0, x)

@njit(fastmath=True)
def softmax_core(x):
    exp_x = np.exp(x)
    # Replace keepdims=True with a reshape
    sum_exp = np.sum(exp_x, axis=1).reshape(-1, 1)
    return exp_x / sum_exp

@njit(fastmath=True)
def forward_prop_core(X, W1, b1, W2, b2, W3, b3):
    """
    Perform forward propagation for a neural network with three layers.

    Parameters:
    X (ndarray): Input data of shape (n_samples, n_features).
    W1 (ndarray): Weight matrix for the first layer of shape (n_features, n_hidden1).
    b1 (ndarray): Bias vector for the first layer of shape (n_hidden1,).
    W2 (ndarray): Weight matrix for the second layer of shape (n_hidden1, n_hidden2).
    b2 (ndarray): Bias vector for the second layer of shape (n_hidden2,).
    W3 (ndarray): Weight matrix for the output layer of shape (n_hidden2, n_classes).
    b3 (ndarray): Bias vector for the output layer of shape (n_classes,).

    Returns:
    a3 (ndarray): Output probabilities of shape (n_samples, n_classes).
    z1 (ndarray): Intermediate values of the first layer of shape (n_samples, n_hidden1).
    a1 (ndarray): Activations of the first layer of shape (n_samples, n_hidden1).
    z2 (ndarray): Intermediate values of the second layer of shape (n_samples, n_hidden2).
    a2 (ndarray): Activations of the second layer of shape (n_samples, n_hidden2).
    z3 (ndarray): Intermediate values of the output layer of shape (n_samples, n_classes).
    """
    # Compute layer 1
    z1 = np.dot(X, W1) + b1
    a1 = relu_core(z1)
    # Compute layer 2
    z2 = np.dot(a1, W2) + b2
    a2 = relu_core(z2)
    # Compute output layer
    z3 = np.dot(a2, W3) + b3
    a3 = softmax_core(z3)
    return a3, z1, a1, z2, a2, z3

@njit(fastmath=True)
def cross_entropy_loss_core(y, y_hat):
    return -np.sum(y * np.log(y_hat)) / y.shape[0]

class NeuralNetwork:
    """
    A class representing a neural network.

    Parameters:
    - input_size (int): The number of input nodes.
    - hidden_size (int): The number of hidden nodes.
    - output_size (int): The number of output nodes.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_params(input_size, hidden_size, output_size)

    def He_initialization(self, input_size, hidden_size):
        """
        He initialization for weight matrices.

        Parameters:
        - input_size (int): The number of input nodes.
        - hidden_size (int): The number of hidden nodes.

        Returns:
        - numpy.ndarray: The initialized weight matrix.
        """
        return np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
    
    def init_params(self, input_size, hidden_size, output_size):
        """
        Initialize the parameters of the neural network.

        Parameters:
        - input_size (int): The number of input nodes.
        - hidden_size (int): The number of hidden nodes.
        - output_size (int): The number of output nodes.
        """
        self.W1 = self.He_initialization(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = self.He_initialization(hidden_size, hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = self.He_initialization(hidden_size, output_size)
        self.b3 = np.zeros((1, output_size))
    
    def warmup(self):
        """
        Warm up the JIT compiler by calling the core functions with dummy inputs.
        """
        relu_core(np.zeros((1, self.hidden_size)))
        softmax_core(np.zeros((1, self.output_size)))
        forward_prop_core(np.zeros((1, self.input_size)), self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        cross_entropy_loss_core(np.zeros((1, self.output_size)), np.zeros((1, self.output_size)))

    def forward_propagation(self, X):
        """
        Perform forward propagation through the neural network.

        Parameters:
        - X (numpy.ndarray): The input data.

        Returns:
        - numpy.ndarray: The output of the neural network.
        """
        self.a3, self.z1, self.a1, self.z2, self.a2, self.z3 = forward_prop_core(
            X, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3
        )
        return self.a3
    
    def cross_entropy_loss(self, y, y_hat):
        """
        Compute the cross-entropy loss.

        Parameters:
        - y (numpy.ndarray): The true labels.
        - y_hat (numpy.ndarray): The predicted labels.

        Returns:
        - float: The cross-entropy loss.
        """
        return cross_entropy_loss_core(y, y_hat)
    
    def compute_gradients(self, X, y):
        """
        Compute the gradients of the parameters.

        Parameters:
        - X (numpy.ndarray): The input data.
        - y (numpy.ndarray): The true labels.

        Returns:
        - list: The gradients of the parameters.
        """
        m = X.shape[0]
        dz3 = self.a3 - y
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        dz2 = np.dot(dz3, self.W3.T) * (self.a2 > 0)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = np.dot(dz2, self.W2.T) * (self.a1 > 0)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        return [dW1, db1, dW2, db2, dW3, db3]
    
    def train(self, X, y, learning_rate, epochs, verbose=True):
        """
        Train the neural network.

        Parameters:
        - X (numpy.ndarray): The input data.
        - y (numpy.ndarray): The true labels.
        - learning_rate (float): The learning rate.
        - epochs (int): The number of training epochs.
        - verbose (bool): Whether to display progress.

        Raises:
        - ImportError: If matplotlib and IPython are not installed.

        Returns:
        - list: The loss values over epochs.
        """
        parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        adam = AdamOptimizer(parameters, learning_rate=learning_rate)
        self.loss_values = []

        if verbose:
            with tqdm(range(epochs), desc="Training", unit="epoch") as pbar:
                for i in pbar:
                    y_hat = self.forward_propagation(X)
                    loss = self.cross_entropy_loss(y, y_hat)
                    self.loss_values.append(loss)
                    
                    gradients = self.compute_gradients(X, y)
                    adam.update(gradients)
                    self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = adam.parameters
                    pbar.set_postfix(loss=f'{loss:.4f}')
        else:
            for i in range(epochs):
                y_hat = self.forward_propagation(X)
                loss = self.cross_entropy_loss(y, y_hat)
                self.loss_values.append(loss)
                
                gradients = self.compute_gradients(X, y)
                adam.update(gradients)
                self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = adam.parameters
            
    def predict(self, X):
        """
        Make predictions using the trained neural network.

        Parameters:
        - X (numpy.ndarray): The input data.

        Returns:
        - numpy.ndarray: The predicted labels.
        """
        return np.argmax(self.forward_propagation(X), axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate the performance of the neural network.

        Parameters:
        - X (numpy.ndarray): The input data.
        - y (numpy.ndarray): The true labels.

        Returns:
        - float: The accuracy of the neural network.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == np.argmax(y, axis=1)) * 100
    
    def save_model(self, file_name):
        """
        Save the model parameters to a file.

        Parameters:
        - file_name (str): The name of the file.
        """
        np.save(file_name, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
    
    def load_model(self, file_name):
        """
        Load the model parameters from a file.

        Parameters:
        - file_name (str): The name of the file.
        """
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = np.load(file_name, allow_pickle=True)
    
    def __repr__(self):
        return f"NeuralNetwork(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size})"
    
    def __str__(self):
        return f"NeuralNetwork with {self.input_size} input nodes, {self.hidden_size} hidden nodes, and {self.output_size} output nodes"
    
    def __len__(self):
        return self.hidden_size
    
    def plot_loss(self, filename="loss_plot-Numba", format="png", show_inline=False, save=True):
        """
        Plot the loss values over epochs.

        Parameters:
        - filename (str): The name of the plot file.
        - format (str): The format of the plot file.
        - show_inline (bool): Whether to display the plot inline.
        - save (bool): Whether to save the plot.

        Raises:
        - ImportError: If matplotlib and IPython are not installed.
        """
        try:
            import matplotlib.pyplot as plt # type: ignore
            from IPython.display import Image, display # type: ignore
        except ImportError as e:
            raise ImportError("matplotlib and IPython are required. Install with 'pip install matplotlib ipython'.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_values)
        #plt.scatter(range(len(self.loss_values)), self.loss_values, color='r')
        plt.title("Loss over epochs (Numba Model)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        if save:
            plt.savefig(f"{filename}.{format}", format=format)
