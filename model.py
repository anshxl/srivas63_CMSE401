#Set up feed-forward neural network with 2 hidden layers
#Use ReLU activation function for hidden layers
#Use softmax activation function for output layer
#Use cross-entropy loss function
#Use Adam optimizer
import numpy as np
from adam import AdamOptimizer
import time
from tqdm import tqdm  # or from tqdm.notebook import tqdm if in a notebook

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_params(input_size, hidden_size, output_size)

    def init_params(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * 0.01
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

    def train(self, X, y, hidden_size, output_size, learning_rate, epochs):
        self.init_params(X.shape[1], hidden_size, output_size)
        parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        adam = AdamOptimizer(parameters, learning_rate=learning_rate)
        
        start_time = time.time()
        with tqdm(range(epochs), desc="Training", unit="epoch") as pbar:
            for i in pbar:
                y_hat = self.forward_propagation(X)
                loss = self.cross_entropy_loss(y, y_hat)
                
                # Compute gradients and update weights
                gradients = self.compute_gradients(X, y)
                adam.update(gradients)
                self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = adam.parameters
                
                # Update the progress bar with the current loss
                pbar.set_postfix(loss=f'{loss:.4f}')
        
        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f} seconds")

        
    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == np.argmax(y, axis=1))*100
    
    def save_model(self, file_name):
        np.save(file_name, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])

    def load_model(self, file_name):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = np.load(file_name, allow_pickle=True)