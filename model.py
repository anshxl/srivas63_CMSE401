import random
import math
from tqdm import tqdm
import time
random.seed(42)

#Import Linear Algebra class
from LinearAlgebra import LinAlg as la
la = la()
#Import Adam Optimizer
from numba_adam import AdamOptimizer

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_params(input_size, hidden_size, output_size)

    def He_initialization(self, input_size, hidden_size):
        scale = math.sqrt(2 / input_size)
        matrix = [
            [random.gauss(0, 1)*scale for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        return matrix
    
    def init_params(self, input_size, hidden_size, output_size):
        self.W1 = self.He_initialization(input_size, hidden_size)
        self.b1 = [0] * hidden_size
        self.W2 = self.He_initialization(hidden_size, hidden_size)
        self.b2 = [0] * hidden_size
        self.W3 = self.He_initialization(hidden_size, output_size)
        self.b3 = [0] * output_size
        
    def relu(self, x):
        # If x is iterable (e.g. a list), apply ReLU to each element; otherwise, treat it as a scalar.
        try:
            return [xi if xi > 0 else 0 for xi in x]
        except TypeError:
            return x if x > 0 else 0

    def _softmax(self, x):
        # Numerically stable softmax for a 1D list.
        max_val = max(x)
        exps = [math.exp(i - max_val) for i in x]
        sum_exps = sum(exps)
        return [i / sum_exps for i in exps]
    
    def softmax(self, x):
        # If x is a batch (a list of lists), apply softmax to each row.
        if isinstance(x[0], list):
            return [self._softmax(row) for row in x]
        else:
            return self._softmax(x)


    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-9
        total_loss = 0
        batch_size = len(y_true)
        for y_t, y_p in zip(y_true, y_pred):
            loss = -sum([y * math.log(y_hat + epsilon) for y, y_hat in zip(y_t, y_p)])
            total_loss += loss
        return total_loss / batch_size
    
    def forward(self, X):        
        # Compute layer 1: z1 = X dot transpose(W1) + b1
        self.z1 = la.dot(X, la.transpose(self.W1))
        self.z1 = la.add_matrices(self.z1, self.b1)
        self.a1 = [self.relu(z) for z in self.z1]
        
        # Compute layer 2: z2 = a1 dot transpose(W2) + b2
        self.z2 = la.dot(self.a1, la.transpose(self.W2))
        self.z2 = la.add_matrices(self.z2, self.b2)
        self.a2 = [self.relu(z) for z in self.z2]
        
        # Compute output layer: z3 = a2 dot transpose(W3) + b3
        self.z3 = la.dot(self.a2, la.transpose(self.W3))
        self.z3 = la.add_matrices(self.z3, self.b3)
        self.a3 = self.softmax(self.z3)
        
        return self.a3

    def compute_gradients(self, X, y_true):
        batch_size = len(X)
        dL_dz3 = la.subtract_matrices(self.a3, y_true)
        dL_dW3 = la.dot(la.transpose(dL_dz3), self.a2)
        dL_dW3 = la.scalar_multiply(1/batch_size, dL_dW3)
        dL_db3 = dL_dz3
        dL_db3 = la.scalar_multiply(1/batch_size, la.sum_axis0(dL_db3)[0])
        dL_da2 = la.dot(dL_dz3, self.W3)
        dL_dz2 = [[dL_da2[i][j] if self.z2[i][j] > 0 else 0 for j in range(self.hidden_size)] for i in range(batch_size)]
        dL_dW2 = la.dot(la.transpose(dL_dz2), self.a1)
        dL_dW2 = la.scalar_multiply(1/batch_size, dL_dW2)
        dL_db2 = dL_dz2
        dL_db2 = la.scalar_multiply(1/batch_size, la.sum_axis0(dL_db2)[0])
        dL_da1 = la.dot(dL_dz2, self.W2)
        dL_dz1 = [[dL_da1[i][j] if self.z1[i][j] > 0 else 0 for j in range(self.hidden_size)] for i in range(batch_size)]
        dL_dW1 = la.dot(la.transpose(dL_dz1), X)
        dL_dW1 = la.scalar_multiply(1/batch_size, dL_dW1)
        dL_db1 = dL_dz1
        dL_db1 = la.scalar_multiply(1/batch_size, la.sum_axis0(dL_db1)[0])
        return [dL_dW1, dL_db1, dL_dW2, dL_db2, dL_dW3, dL_db3]
    
    def update_params(self, gradients, learning_rate):
        # Update weight matrices normally.
        self.W1 = la.subtract_matrices(self.W1, la.scalar_multiply(learning_rate, gradients[0]))
        self.b1 = la.subtract_vectors(self.b1, la.scalar_multiply(learning_rate, gradients[1]))
        self.W2 = la.subtract_matrices(self.W2, la.scalar_multiply(learning_rate, gradients[2]))
        self.b2 = la.subtract_vectors(self.b2, la.scalar_multiply(learning_rate, gradients[3]))
        self.W3 = la.subtract_matrices(self.W3, la.scalar_multiply(learning_rate, gradients[4]))
        self.b3 = la.subtract_vectors(self.b3, la.scalar_multiply(learning_rate, gradients[5]))

    def train(self, X_train, y_train, epochs, learning_rate):
        self.loss_values = []
        start_time = time.time()
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        optimizer = AdamOptimizer(params, learning_rate)
        with tqdm(range(epochs), desc='Training', unit='epoch') as pbar:
            for epoch in pbar:
                total_loss = 0
                # for X, y in zip(X_train, y_train):
                y_pred = self.forward(X_train)
                total_loss += self.cross_entropy_loss(y_train, y_pred)
                self.loss_values.append(total_loss)

                gradients = self.compute_gradients(X_train, y_train)
                # self.update_params(gradients, learning_rate)
                optimizer.update(gradients)
                self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = optimizer.parameters
        
                pbar.set_postfix({'loss': self.loss_values[-1]})
        
        end_time = time.time()
        print(f'Training time: {end_time - start_time:.2f} seconds')
    
    def predict(self, X):
        predictions = self.forward(X)
        return [p.index(max(p)) for p in predictions]
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        y_true = [t.index(1) for t in y]
        correct = sum([1 for p, t in zip(predictions, y_true) if p == t])
        return correct / len(y) * 100
    
    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
    
    def check_params(self):
        return self.params
    
