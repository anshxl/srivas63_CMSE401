from numba.typed import List # type: ignore
from numba import njit, prange # type: ignore
import math

@njit
def zeros_like(self, x):
    result = List()
    if isinstance(x, list):
        for _ in x:
            result.append(0.0)
        return result
    else:
        return 0.0

@njit
def add_vectors(self, x, y):
    result = List()
    for a, b in zip(x, y):
        result.append(a + b)
    return result

@njit
def subtract_vectors(self, x, y):
    result = List()
    for a, b in zip(x, y):
        result.append(a - b)
    return result

@njit
def add_matrices(self, x, y):
    # If both x and y are vectors, add elementwise.
    if not isinstance(x[0], list) and not isinstance(y[0], list):
        return self.add_vectors(x, y)
    # If y is a vector and x a matrix, replicate y into a matrix.
    if not isinstance(y[0], list):
        y_matrix = List()
        for _ in x:
            row = List()
            for val in y:
                row.append(val)
            y_matrix.append(row)
        result = List()
        for x_row, y_row in zip(x, y_matrix):
            result.append(self.add_vectors(x_row, y_row))
        return result
    # Otherwise, both are matrices.
    result = List()
    for x_row, y_row in zip(x, y):
        result.append(self.add_vectors(x_row, y_row))
    return result

@njit
def subtract_matrices(self, x, y):
    if not isinstance(x[0], list) and not isinstance(y[0], list):
        return self.subtract_vectors(x, y)
    if not isinstance(y[0], list):
        y_matrix = List()
        for _ in x:
            row = List()
            for val in y:
                row.append(val)
            y_matrix.append(row)
        result = List()
        for x_row, y_row in zip(x, y_matrix):
            result.append(self.subtract_vectors(x_row, y_row))
        return result
    result = List()
    for x_row, y_row in zip(x, y):
        result.append(self.subtract_vectors(x_row, y_row))
    return result

@njit
def scalar_multiply(self, c, x):
    if isinstance(x, list):
        result = List()
        for elem in x:
            result.append(self.scalar_multiply(c, elem))
        return result
    else:
        return c * x

@njit
def dot(self, x, y):
    n = len(x)
    d = len(x[0])
    m = len(y[0])
    result = List()
    for i in range(n):
        row = List()
        for _ in range(m):
            row.append(0.0)
        result.append(row)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(d):
                s += x[i][k] * y[k][j]
            result[i][j] = s
    return result

@njit
def sum_axis0(self, x):
    if not x:
        return List()
    if not isinstance(x[0], list):
        vec = List()
        for val in x:
            vec.append(val)
        result = List()
        result.append(vec)
        return result
    num_cols = len(x[0])
    sums = List()
    for _ in range(num_cols):
        sums.append(0.0)
    for row in x:
        for j, val in enumerate(row):
            sums[j] = sums[j] + val
    result = List()
    result.append(sums)
    return result

@njit
def matrix_mask(self, x, condition_func):
    result = List()
    for row in x:
        row_result = List()
        for elem in row:
            row_result.append(1 if condition_func(elem) else 0)
        result.append(row_result)
    return result

@njit
def transpose(self, matrix):
    result = List()
    for row in zip(*matrix):
        new_row = List()
        for elem in row:
            new_row.append(elem)
        result.append(new_row)
    return result
