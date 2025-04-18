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
    A class representing a neural network model optimized for GPU computation using NumPy.

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
        He initialization for random values and square root.

        Parameters:
        - input_size (int): The number of input nodes.
        - hidden_size (int): The number of hidden nodes.

        Returns:
        - np.ndarray: Initialized weight matrix.
        """
        return np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
    
    def init_params(self, input_size, hidden_size, output_size):
        """
        Initializes the parameters of the neural network model.

        Args:
            input_size (int): The size of the input layer.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer.

        Returns:
            None
        """
        self.W1 = self.He_initialization(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = self.He_initialization(hidden_size, hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = self.He_initialization(hidden_size, output_size)
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        """
        Applies the Rectified Linear Unit (ReLU) activation function to the input.

        Parameters:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The output array after applying the ReLU activation function.
        """
        return np.maximum(0, x)
    
    def softmax(self, x):
        """
        Compute the softmax function for the given input array.

        Parameters:
        x (numpy.ndarray): Input array.

        Returns:
        numpy.ndarray: Softmax output array.

        """
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y, y_hat):
        """
        Calculates the cross-entropy loss between the predicted values (y_hat) and the true values (y).

        Parameters:
            y (numpy.ndarray): True values.
            y_hat (numpy.ndarray): Predicted values.

        Returns:
            float: Cross-entropy loss.

        """
        return -np.sum(y * np.log(y_hat)) / y.shape[0]

    def forward_propagation(self, X):
        """
        Perform forward propagation to compute the output of the neural network.

        Args:
            X (ndarray): Input data of shape (batch_size, input_size).

        Returns:
            ndarray: Output of the neural network, a probability distribution over the classes.
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3
    
    def compute_gradients(self, X, y):
        """
        Compute the gradients of the model's parameters with respect to the loss function.

        Args:
            X (numpy.ndarray): Input data of shape (m, n), where m is the number of examples and n is the number of features.
            y (numpy.ndarray): Target values of shape (m,).

        Returns:
            list: A list containing the gradients of the model's parameters in the following order: dW1, db1, dW2, db2, dW3, db3.
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
        Trains the model using the given input data and labels.

        Parameters:
            X (numpy.ndarray): Input data of shape (num_samples, num_features).
            y (numpy.ndarray): Target labels of shape (num_samples, num_classes).
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            verbose (bool, optional): Whether to display progress bar and loss updates. Defaults to True.

        Returns:
            None
        """
        # self.init_params(X.shape[1], hidden_size, output_size)
        parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        adam = AdamOptimizer(parameters, learning_rate=learning_rate)

        # Initialize list to store loss values
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
        """
        Predicts the class labels for the given input data.

        Parameters:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted class labels of shape (n_samples,).
        """
        return np.argmax(self.forward_propagation(X), axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluates the model's performance on the given input data.

        Parameters:
        X (array-like): The input data.
        y (array-like): The target labels.

        Returns:
        float: The accuracy of the model, expressed as a percentage.

        """
        y_pred = self.predict(X)
        return np.mean(y_pred == np.argmax(y, axis=1))*100
    
    def save_model(self, file_name):
        """
        Save the model parameters to a file.

        Parameters:
        - file_name (str): The name of the file to save the model parameters to.

        Returns:
        None
        """
        np.save(file_name, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

    def load_model(self, file_name):
        """
        Loads the model parameters from a file.

        Parameters:
        - file_name (str): The name of the file to load the model parameters from.

        Returns:
        None
        """
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = np.load(file_name, allow_pickle=True)
    
    def __repr__(self):
        """
        Returns a string representation of the NeuralNetwork object.
        """
        return f"NeuralNetwork(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size})"

    def __str__(self):
        """
        Returns a string representation of the NeuralNetwork object.

        The string includes information about the number of input nodes, hidden nodes, and output nodes.

        Returns:
            str: A string representation of the NeuralNetwork object.
        """
        return f"NeuralNetwork with {self.input_size} input nodes, {self.hidden_size} hidden nodes, and {self.output_size} output nodes"
    
    def __len__(self):
        """
        Returns the length of the object.

        Returns:
            int: The length of the object.
        """
        return self.hidden
    
    def plot_loss(self, filename="loss_plot-NumPy", format="png", show_inline=False, save=True):
        """
        Plots the loss values over epochs and optionally saves the plot as an image file.

        Parameters:
            filename (str): The filename to save the plot as. Default is "loss_plot-NumPy".
            format (str): The format of the image file. Default is "png".
            show_inline (bool): Whether to display the plot inline in Jupyter Notebook. Default is False.
            save (bool): Whether to save the plot as an image file. Default is True.

        Raises:
            ImportError: If matplotlib and IPython are not installed.

        Returns:
            None
        """
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
        Plot the architecture of the model.

        Parameters:
        - filename (str): The name of the file to save the plot.
        - format (str): The format of the saved file (e.g., 'png', 'pdf', 'svg').
        - show_inline (bool): Whether to display the plot inline in a notebook.

        Raises:
        - ImportError: If graphviz and IPython are not installed.

        Returns:
        - None
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


            