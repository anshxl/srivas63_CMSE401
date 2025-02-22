import unittest
import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LinearAlgebra import LinAlg

def test_zeros_like():
    la = LinAlg()
    assert la.zeros_like([1, 2, 3]) == [0, 0, 0], "zeros_like for list failed"
    assert la.zeros_like(5) == 0.0, "zeros_like for scalar failed"

def test_add_vectors():
    la = LinAlg()
    assert la.add_vectors([1, 2, 3], [4, 5, 6]) == [5, 7, 9], "add_vectors for lists failed"
    assert la.add_vectors(3, 4) == 7, "add_vectors for scalars failed"

def test_subtract_vectors():
    la = LinAlg()
    assert la.subtract_vectors([5, 7, 9], [1, 2, 3]) == [4, 5, 6], "subtract_vectors for lists failed"
    assert la.subtract_vectors(10, 3) == 7, "subtract_vectors for scalars failed"

def test_scalar_multiply():
    la = LinAlg()
    assert la.scalar_multiply(2, [1, 2, 3]) == [2, 4, 6], "scalar_multiply for lists failed"
    assert la.scalar_multiply(3, 4) == 12, "scalar_multiply for scalars failed"

def test_elementwise_multiply():
    la = LinAlg()
    assert la.elementwise_multiply([1, 2, 3], [4, 5, 6]) == [4, 10, 18], "elementwise_multiply for lists failed"
    assert la.elementwise_multiply(3, 4) == 12, "elementwise_multiply for scalars failed"

def test_elementwise_divide():
    la = LinAlg()
    assert la.elementwise_divide([10, 20, 30], [2, 5, 3]) == [5, 4, 10], "elementwise_divide for lists failed"
    assert la.elementwise_divide(10, 2) == 5, "elementwise_divide for scalars failed"

def test_square():
    la = LinAlg()
    assert la.square([2, 3, 4]) == [4, 9, 16], "square for lists failed"
    assert la.square(5) == 25, "square for scalars failed"

def test_sqrt():
    la = LinAlg()
    # Allow small rounding differences
    sqrt_list = la.sqrt([4, 9, 16])
    assert all(abs(a - b) < 1e-6 for a, b in zip(sqrt_list, [2, 3, 4])), "sqrt for lists failed"
    assert abs(la.sqrt(16) - 4) < 1e-6, "sqrt for scalars failed"

def test_dot():
    la = LinAlg()
    # Test with 2x2 matrices.
    x = [[1, 2], [3, 4]]
    y = [[5, 6], [7, 8]]
    result = la.dot(x, y)
    expected = [[1*5 + 2*7, 1*6 + 2*8],
                [3*5 + 4*7, 3*6 + 4*8]]  # [[19, 22], [43, 50]]
    assert result == expected, f"dot product failed: {result} != {expected}"

def test_sum_axis0():
    la = LinAlg()
    # Test with a 2D list.
    matrix = [[1, 2, 3], [4, 5, 6]]
    result = la.sum_axis0(matrix)
    expected = [[5, 7, 9]]
    assert result == expected, f"sum_axis0 for 2D list failed: {result} != {expected}"
    
    # Test with a 1D list.
    vec = [1, 2, 3]
    result = la.sum_axis0(vec)
    expected = [[1, 2, 3]]
    assert result == expected, f"sum_axis0 for 1D list failed: {result} != {expected}"

def test_matrix_mask():
    la = LinAlg()
    matrix = [[-1, 2, -3], [4, -5, 6]]
    # Condition: value > 0
    result = la.matrix_mask(matrix, lambda x: x > 0)
    expected = [[0, 1, 0], [1, 0, 1]]
    assert result == expected, f"matrix_mask failed: {result} != {expected}"

def test_subtract_vectors():
    la = LinAlg()
    v1 = [10, 20, 30]
    v2 = [1, 2, 3]
    result = la.subtract_vectors(v1, v2)
    expected = [9, 18, 27]
    assert result == expected, f"subtract_vectors failed: {result} != {expected}"

def test_subtract_matrices():
    la = LinAlg()
    A = [[10, 20, 30], [40, 50, 60]]
    B = [[1, 2, 3], [4, 5, 6]]
    # Assume subtract_matrices is implemented as:
    # def subtract_matrices(self, A, B):
    #     return [self.subtract_vectors(rowA, rowB) for rowA, rowB in zip(A, B)]
    result = la.subtract_matrices(A, B)
    expected = [[9, 18, 27], [36, 45, 54]]
    assert result == expected, f"subtract_matrices failed: {result} != {expected}"

def test_transpose():
    la = LinAlg()
    A = [[1, 2, 3], [4, 5, 6]]
    result = la.transpose(A)
    expected = [[1, 4], [2, 5], [3, 6]]
    assert result == expected, f"transpose failed: {result} != {expected}"

def run_all_tests():
    test_zeros_like()
    test_add_vectors()
    test_subtract_vectors()
    test_scalar_multiply()
    test_elementwise_multiply()
    test_elementwise_divide()
    test_square()
    test_sqrt()
    test_dot()
    test_sum_axis0()
    test_matrix_mask()
    test_subtract_vectors()
    test_subtract_matrices()
    test_transpose()
    print("All LinearAlegbra tests passed.")

if __name__ == "__main__":
    run_all_tests()