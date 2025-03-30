import numpy as np
import time
from numba import jit #type: ignore

# Pure Python loop: computes the sum of sine values
def sum_loop(n):
    total = 0.0
    for i in range(n):
        total += np.sin(i)
    return total

# Numba optimized loop: uses JIT compilation for acceleration
@jit(nopython=True)
def sum_loop_numba(n):
    total = 0.0
    for i in range(n):
        total += np.sin(i)
    return total

if __name__ == "__main__":
    n = 10_000_000  # Number of iterations

    # Measure performance of the pure Python loop
    start = time.time()
    result_py = sum_loop(n)
    time_py = time.time() - start
    print(f"Pure Python result: {result_py:.3f}, Time: {time_py:.3f} seconds\n")

    # Measure performance of the Numba optimized loop
    start = time.time()
    result_nb = sum_loop_numba(n)
    time_nb = time.time() - start
    print(f"Numba optimized result: {result_nb:.3f} Time: {time_nb:.3f} seconds")
