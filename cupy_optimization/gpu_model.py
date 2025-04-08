# GPU MLP
# Neural Network implemented using CuPy
import cupy as cp #type: ignore[import]
from tqdm import tqdm #type: ignore[import]
from gpu_adam import GPUAdamOptimizer  # Import the GPU Adam optimizer

class GPUNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_params(input_size, hidden_size, output_size)

    def He_initialization(self, input_size, hidden_size):
        # He initialization using CuPy for random values and square root
        return cp.random.randn(input_size, hidden_size) * cp.sqrt(2 / input_size)
    
    def init_params(self, input_size, hidden_size, output_size):
        self.W1 = self.He_initialization(input_size, hidden_size)
        self.b1 = cp.zeros((1, hidden_size))
        self.W2 = self.He_initialization(hidden_size, hidden_size)
        self.b2 = cp.zeros((1, hidden_size))
        self.W3 = self.He_initialization(hidden_size, output_size)
        self.b3 = cp.zeros((1, output_size))

    def relu(self, x):
        return cp.maximum(0, x)
    
    def softmax(self, x):
        exp_x = cp.exp(x)
        return exp_x / cp.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y, y_hat):
        return -cp.sum(y * cp.log(y_hat)) / y.shape[0]

    def forward_propagation(self, X):
        self.z1 = cp.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = cp.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = cp.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def compute_gradients(self, X, y):
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

    def train(self, X, y, learning_rate, epochs):
        # Ensure that X and y are on the GPU
        # (If they are NumPy arrays, convert with cp.asarray)
        X_gpu = X if isinstance(X, cp.ndarray) else cp.asarray(X)
        y_gpu = y if isinstance(y, cp.ndarray) else cp.asarray(y)
        
        parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        adam = GPUAdamOptimizer(parameters, learning_rate=learning_rate)
        self.loss_values = []
        
        # Train without logging loss for each epoch on the progress bar
        for _ in tqdm(range(epochs), desc="Training", unit="epoch"):
            y_hat = self.forward_propagation(X_gpu)
            loss = self.cross_entropy_loss(y_gpu, y_hat)
            self.loss_values.append(loss)
            
            gradients = self.compute_gradients(X_gpu, y_gpu)
            adam.update(gradients)
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = adam.parameters

    def predict(self, X):
        # Ensure that the input is a CuPy array
        X_gpu = X if isinstance(X, cp.ndarray) else cp.asarray(X)
        return cp.argmax(self.forward_propagation(X_gpu), axis=1)
    
    def evaluate(self, X, y):
        # Ensure that inputs are on the GPU
        X_gpu = X if isinstance(X, cp.ndarray) else cp.asarray(X)
        y_gpu = y if isinstance(y, cp.ndarray) else cp.asarray(y)
        y_pred = self.predict(X_gpu)
        return cp.mean(y_pred == cp.argmax(y_gpu, axis=1)) * 100