# GPU MLP
# Neural Network implemented using CuPy
import cupy as cp #type: ignore[import]
from tqdm import tqdm #type: ignore[import]
from cupy_optimization.gpu_adam import GPUAdamOptimizer  # Import the GPU Adam optimizer

class GPUNeuralNetwork:
    """
    A class representing a neural network model optimized for GPU computation using CuPy.

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
        He initialization using CuPy for random values and square root.

        Parameters:
        - input_size (int): The number of input nodes.
        - hidden_size (int): The number of hidden nodes.

        Returns:
        - cp.ndarray: Initialized weight matrix.
        """
        return cp.random.randn(input_size, hidden_size) * cp.sqrt(2 / input_size)
    
    def init_params(self, input_size, hidden_size, output_size):
        """
        Initialize the parameters of the neural network.

        Parameters:
        - input_size (int): The number of input nodes.
        - hidden_size (int): The number of hidden nodes.
        - output_size (int): The number of output nodes.
        """
        self.W1 = self.He_initialization(input_size, hidden_size)
        self.b1 = cp.zeros((1, hidden_size))
        self.W2 = self.He_initialization(hidden_size, hidden_size)
        self.b2 = cp.zeros((1, hidden_size))
        self.W3 = self.He_initialization(hidden_size, output_size)
        self.b3 = cp.zeros((1, output_size))

    def relu(self, x):
        """
        Rectified Linear Unit (ReLU) activation function.

        Parameters:
        - x (cp.ndarray): Input array.

        Returns:
        - cp.ndarray: Output array after applying ReLU.
        """
        return cp.maximum(0, x)
    
    def softmax(self, x):
        """
        Softmax activation function.

        Parameters:
        - x (cp.ndarray): Input array.

        Returns:
        - cp.ndarray: Output array after applying softmax.
        """
        exp_x = cp.exp(x)
        return exp_x / cp.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y, y_hat):
        """
        Compute the cross-entropy loss.

        Parameters:
        - y (cp.ndarray): True labels.
        - y_hat (cp.ndarray): Predicted labels.

        Returns:
        - float: Cross-entropy loss.
        """
        return -cp.sum(y * cp.log(y_hat)) / y.shape[0]

    def forward_propagation(self, X):
        """
        Perform forward propagation through the neural network.

        Parameters:
        - X (cp.ndarray): Input data.

        Returns:
        - cp.ndarray: Output predictions.
        """
        self.z1 = cp.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = cp.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = cp.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def compute_gradients(self, X, y):
        """
        Compute the gradients of the parameters.

        Parameters:
        - X (cp.ndarray): Input data.
        - y (cp.ndarray): True labels.

        Returns:
        - list: Gradients of the parameters.
        """
        m = X.shape[0]
        dz3 = self.a3 - y
        dW3 = cp.dot(self.a2.T, dz3) / m
        db3 = cp.sum(dz3, axis=0, keepdims=True) / m
        dz2 = cp.dot(dz3, self.W3.T) * (self.a2 > 0)
        dW2 = cp.dot(self.a1.T, dz2) / m
        db2 = cp.sum(dz2, axis=0, keepdims=True) / m
        dz1 = cp.dot(dz2, self.W2.T) * (self.a1 > 0)
        dW1 = cp.dot(X.T, dz1) / m
        db1 = cp.sum(dz1, axis=0, keepdims=True) / m
        return [dW1, db1, dW2, db2, dW3, db3]

    def train(self, X, y, learning_rate, epochs, verbose=True):
        """
        Train the neural network.

        Parameters:
        - X (cp.ndarray): Input data.
        - y (cp.ndarray): True labels.
        - learning_rate (float): Learning rate for optimization.
        - epochs (int): Number of training epochs.
        - verbose (bool): Whether to display progress bar and loss values.

        Raises:
        - ImportError: If matplotlib and IPython are not installed.

        Note:
        - If X and y are NumPy arrays, they will be converted to CuPy arrays.
        """
        # Ensure that X and y are on the GPU
        # (If they are NumPy arrays, convert with cp.asarray)
        X_gpu = X if isinstance(X, cp.ndarray) else cp.asarray(X)
        y_gpu = y if isinstance(y, cp.ndarray) else cp.asarray(y)
        
        parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        adam = GPUAdamOptimizer(parameters, learning_rate=learning_rate)
        self.loss_values = []
        
        # Train without logging loss for each epoch on the progress bar
        if verbose:
            for _ in tqdm(range(epochs), desc="Training", unit="epoch"):
                y_hat = self.forward_propagation(X_gpu)
                loss = self.cross_entropy_loss(y_gpu, y_hat)
                self.loss_values.append(loss)
                
                gradients = self.compute_gradients(X_gpu, y_gpu)
                adam.update(gradients)
                self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = adam.parameters
        else:
            for _ in range(epochs):
                y_hat = self.forward_propagation(X_gpu)
                loss = self.cross_entropy_loss(y_gpu, y_hat)
                self.loss_values.append(loss)
                
                gradients = self.compute_gradients(X_gpu, y_gpu)
                adam.update(gradients)
                self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = adam.parameters

    def predict(self, X):
        """
        Make predictions using the trained neural network.

        Parameters:
        - X (cp.ndarray): Input data.

        Returns:
        - cp.ndarray: Predicted labels.
        """
        # Ensure that the input is a CuPy array
        X_gpu = X if isinstance(X, cp.ndarray) else cp.asarray(X)
        return cp.argmax(self.forward_propagation(X_gpu), axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate the performance of the neural network.

        Parameters:
        - X (cp.ndarray): Input data.
        - y (cp.ndarray): True labels.

        Returns:
        - float: Accuracy of the predictions.
        """
        # Ensure that inputs are on the GPU
        X_gpu = X if isinstance(X, cp.ndarray) else cp.asarray(X)
        y_gpu = y if isinstance(y, cp.ndarray) else cp.asarray(y)
        y_pred = self.predict(X_gpu)
        return cp.mean(y_pred == cp.argmax(y_gpu, axis=1)) * 100
    
    def __repr__(self):
        return f"NeuralNetwork(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size})"
    
    def __str__(self):
        return f"NeuralNetwork with {self.input_size} input nodes, {self.hidden_size} hidden nodes, and {self.output_size} output nodes"
    
    def __len__(self):
        return self.hidden_size
    
    def plot_loss(self, filename="GPU_loss_plot", format="png", show_inline=False, save=True):
        """
        Plot the loss values over epochs.

        Parameters:
        - filename (str): Name of the plot file.
        - format (str): Format of the plot file.
        - show_inline (bool): Whether to display the plot inline.
        - save (bool): Whether to save the plot file.

        Raises:
        - ImportError: If matplotlib and IPython are not installed.
        """
        try:
            import matplotlib.pyplot as plt # type: ignore
            from IPython.display import Image, display # type: ignore
        except ImportError as e:
            raise ImportError("matplotlib and IPython are required. Install with 'pip install matplotlib ipython'.")
        
        plt.figure(figsize=(10, 6))
        # Convert loss_values from CuPy to NumPy
        loss_values_np = cp.asnumpy(cp.array(self.loss_values))
        plt.plot(loss_values_np)
        plt.title("Loss over epochs - GPU Model")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        if save:
            plt.savefig(f"{filename}.{format}", format=format)
    