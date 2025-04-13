#Set up feed-forward neural network with 2 hidden layers
#Use ReLU activation function for hidden layers
#Use softmax activation function for output layer
#Use cross-entropy loss function
#Use Adam optimizer
import numpy as np
import time
import sys
import os
from tqdm import tqdm  # type: ignore

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adam import AdamOptimizer


class NeuralNetwork:
    """
    A class representing a neural network model.

    Parameters:
    - input_size (int): The number of input nodes.
    - hidden_size (int): The number of nodes in each hidden layer.
    - output_size (int): The number of output nodes.

    Methods:
    - He_initialization(input_size, hidden_size): Initializes the weights using He initialization.
    - init_params(input_size, hidden_size, output_size): Initializes the parameters of the neural network.
    - relu(x): Applies the ReLU activation function to the input.
    - softmax(x): Applies the softmax activation function to the input.
    - cross_entropy_loss(y, y_hat): Computes the cross-entropy loss between the predicted and actual values.
    - forward_propagation(X): Performs forward propagation to compute the output of the neural network.
    - compute_gradients(X, y): Computes the gradients of the parameters with respect to the loss.
    - train(X, y, learning_rate, epochs, verbose=True): Trains the neural network on the given data.
    - predict(X): Predicts the output for the given input.
    - evaluate(X, y): Evaluates the accuracy of the neural network on the given data.
    - save_model(file_name): Saves the model parameters to a file.
    - load_model(file_name): Loads the model parameters from a file.
    - plot_loss(filename="loss_plot", format="png", show_inline=False, save=True): Plots the loss over epochs.
    - plot_architecture(filename="model_architecture", format="png", show_inline=False): Plots the model architecture.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_params(input_size, hidden_size, output_size)

    def He_initialization(self, input_size, hidden_size):
        return np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
    
    def init_params(self, input_size, hidden_size, output_size):
        self.W1 = self.He_initialization(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = self.He_initialization(hidden_size, hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = self.He_initialization(hidden_size, output_size)
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y, y_hat):
        return -np.sum(y * np.log(y_hat)) / y.shape[0]

    def forward_propagation(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3
    
    def compute_gradients(self, X, y):
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
        # self.init_params(X.shape[1], hidden_size, output_size)
        parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        adam = AdamOptimizer(parameters, learning_rate=learning_rate)

        #Initialize list to store loss values
        self.loss_values = []
        
        if verbose:
            with tqdm(range(epochs), desc="Training", unit="epoch") as pbar:
                for i in pbar:
                    y_hat = self.forward_propagation(X)
                    loss = self.cross_entropy_loss(y, y_hat)
                    self.loss_values.append(loss)

                    # Compute gradients and update weights
                    gradients = self.compute_gradients(X, y)
                    adam.update(gradients)
                    self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = adam.parameters
                    
                    # Update the progress bar with the current loss
                    pbar.set_postfix(loss=f'{loss:.4f}')
        else:
            for i in range(epochs):
                y_hat = self.forward_propagation(X)
                loss = self.cross_entropy_loss(y, y_hat)
                self.loss_values.append(loss)

                # Compute gradients and update weights
                gradients = self.compute_gradients(X, y)
                adam.update(gradients)
                self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = adam.parameters
        
    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == np.argmax(y, axis=1))*100
    
    def save_model(self, file_name):
        np.save(file_name, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

    def load_model(self, file_name):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = np.load(file_name, allow_pickle=True)
    
    def __repr__(self):
        return f"NeuralNetwork(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size})"

    def __str__(self):
        return f"NeuralNetwork with {self.input_size} input nodes, {self.hidden_size} hidden nodes, and {self.output_size} output nodes"
    
    def __len__(self):
        return self.hidden
    
    def plot_loss(self, filename="loss_plot-NumPy", format="png", show_inline=False, save=True):
        try:
            import matplotlib.pyplot as plt #type: ignore
            from IPython.display import Image, display #type: ignore
        except ImportError as e:
            raise ImportError("matplotlib and IPython are required. Install with 'pip install matplotlib ipython'.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_values)
        #plt.scatter(range(len(self.loss_values)), self.loss_values, color='r')
        plt.title("Loss over epochs - Baseline Model")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        if save:
            plt.savefig(f"{filename}.{format}", format=format)
            
    def plot_architecture(self, filename="model_architecture", format="png", show_inline=False):
        """
        Plots the model architecture and saves it to a file.
        
        If running in a Jupyter Notebook, it will also display the plot inline.
        
        Requires:
            - graphviz: Install via pip install graphviz
            - IPython: Typically available in Jupyter environments.
        """
        try:
            from graphviz import Digraph #type: ignore
            from IPython.display import Image, display #type: ignore
        except ImportError as e:
            raise ImportError("graphviz and IPython are required. Install with 'pip install graphviz ipython'.")

        dot = Digraph(name="Model Architecture", format=format)
        dot.attr(rankdir='LR', splines='ortho')
        
        # Create nodes for each layer
        dot.node("input", f"Input\n({self.input_size})")
        dot.node("hidden1", f"Hidden Layer 1\n({self.hidden_size})\nReLU")
        dot.node("hidden2", f"Hidden Layer 2\n({self.hidden_size})\nReLU")
        dot.node("output", f"Output\n({self.output_size})\nSoftmax")
        
        # Connect layers with edges indicating parameters
        dot.edge("input", "hidden1", label="W1, b1")
        dot.edge("hidden1", "hidden2", label="W2, b2")
        dot.edge("hidden2", "output", label="W3, b3")
        
        # Render the graph to a file
        dot.render(filename, format=format, cleanup=True)
        
        # Optionally display inline (if in a notebook)
        if show_inline:
            display(Image(dot.pipe(format=format)))


            