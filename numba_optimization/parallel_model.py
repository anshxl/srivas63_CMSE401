import numpy as np
from numba import njit, prange # type: ignore
from tqdm import tqdm
import time
from adam import AdamOptimizer

@njit
def relu(x):
    return np.maximum(x, 0)

@njit
def softmax(x):
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x)
    return exp_x / sum_exp

@njit(parallel=True)
def batch_forward_prop(X_batch, W1, b1, W2, b2, W3, b3):
    n_samples = X_batch.shape[0]
    outputs = np.empty((n_samples, W3.shape[1]))
    #Loop over each sample in the batch in parallel
    for i in prange(n_samples):
        #Forward pass
        z1 = np.dot(X_batch[i], W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = relu(z2)
        z3 = np.dot(a2, W3) + b3
        a3 = softmax(z3)
        outputs[i] = a3
    return outputs

@njit
def cross_entropy_loss_numba(y, y_hat):
    m = y.shape[0]
    # Compute loss averaged over batch
    loss = -np.sum(y * np.log(y_hat)) / m
    return loss

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._init_params(input_size, hidden_size, output_size)

    def He_initialization(self, size1, size2):
        return np.random.randn(size1, size2) * np.sqrt(2/size1)
    
    def _init_params(self, input_size, hidden_size, output_size):
        self.W1 = self.He_initialization(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = self.He_initialization(hidden_size, hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = self.He_initialization(hidden_size, output_size)
        self.b3 = np.zeros(output_size)

    def warm_up(self, X):
        _ = batch_forward_prop(X[:2], self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        _ = relu(np.array([1.0, -1.0]))
        _ = softmax(np.array([1.0, 2.0]))

    def forward(self, X):
        return batch_forward_prop(X, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)

    def train(self, X, epochs, batch_size):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size

        start_time = time.time()
        preds = []
        for epoch in range(epochs):
            pred = []
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = X[start:end]
                y_hat = self.forward(X_batch)
                predictions = np.argmax(y_hat, axis=1)
                preds.append(predictions)
            preds.append(pred)
            print(f"Epoch {epoch+1}/{epochs}")
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")
        return preds
            