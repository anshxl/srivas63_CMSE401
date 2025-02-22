import math

class LinearAlegbra:

    def __init__(self):
        pass

    def zeros_like(self, x):
        if isinstance(x, list):
            return [0 for _ in x]
        else:
            return 0.0
    
    def add_vectors(self, x, y):
        if isinstance(x, list):
            return [x_i + y_i for x_i, y_i in zip(x, y)]
        else:
            return x + y
    
    def subtract_vectors(self, x, y):
        if isinstance(x, list):
            return [x_i - y_i for x_i, y_i in zip(x, y)]
        else:
            return x - y

    def add_matrices(self, x, y):
        # If y is a vector (1D list), replicate it to match the number of rows in x.
        if not x:
            return []
        if not isinstance(y[0], list):
            # y is a vector; create a matrix where each row is y.
            y_matrix = [y for _ in range(len(x))]
            return [self.add_vectors(x_i, y_i) for x_i, y_i in zip(x, y_matrix)]
        else:
            return [self.add_vectors(x_i, y_i) for x_i, y_i in zip(x, y)]
        
    def subtract_matrices(self, x, y):
        # If y is a vector (1D list), replicate it to match the number of rows in x.
        if not x:
            return []
        if not isinstance(y[0], list):
            # y is a vector; create a matrix where each row is y.
            y_matrix = [y for _ in range(len(x))]
            return [self.subtract_vectors(x_i, y_i) for x_i, y_i in zip(x, y_matrix)]
        else:
            return [self.subtract_vectors(x_i, y_i) for x_i, y_i in zip(x, y)]
    
    def scalar_multiply(self, c, x):
        if isinstance(x, list):
            return [self.scalar_multiply(c, elem) for elem in x]
        else:
            return c * x
    
    def elementwise_multiply(self, x, y):
        if isinstance(x, list):
            return [x_i * y_i for x_i, y_i in zip(x, y)]
        else:
            return x * y
    
    def elementwise_divide(self, x, y):
        if isinstance(x, list):
            return [x_i / y_i for x_i, y_i in zip(x, y)]
        else:
            return x / y
    
    def square(self, x):
        if isinstance(x, list):
            return [x_i**2 for x_i in x]
        else:
            return x**2
    
    def sqrt(self, x):
        if isinstance(x, list):
            return [math.sqrt(x_i) for x_i in x]
        else:
            return math.sqrt(x)
    
    def dot(self, x, y):
        n = len(x)
        d = len(x[0])
        m = len(y[0])

        result = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                s = 0
                for k in range(d):
                    s += x[i][k] * y[k][j]
                result[i][j] = s
        return result
    
    def sum_axis0(self, x):
        if not x:
            return [[]]
        if not isinstance(x[0], list):
            # Assume it's a 1D list.
            return [x]
        num_cols = len(x[0])
        sums = [0] * num_cols
        for row in x:
            for j, val in enumerate(row):
                sums[j] += val
        return [sums]
    
    def matrix_mask(self, x, condition_func):
        return [[1 if condition_func(x) else 0 for x in row] for row in x]
    
    def transpose(self, matrix):
        return [list(row) for row in zip(*matrix)]

    def add_epsilon(self, x, epsilon):
        if isinstance(x, list):
            return [x_i + epsilon for x_i in x]
        else:
            return x + epsilon