import numpy as np
import time
from numba import njit #type: ignore

# Grid
xmin = 0.0
xmax = 10.0
nx = 512
dx = (xmax - xmin) / nx
x = np.linspace(xmin, xmax, nx)

# Time
tmin = 0.0
tmax = 10.0
nt = 1000000
dt = (tmax - tmin) / nt

# Initial Conditions
def initialize_conditions():
    y = np.exp(-(x - 5)**2)  # Initial displacement
    v = np.zeros(nx)         # Initial velocity
    a = np.zeros(nx)         # Initial acceleration
    return y, v, a

# Time Evolution Function
def time_evolution(y, v, a, nt, dx, dt):
    for _ in range(nt):
        a[1:-1] = (y[2:] + y[:-2] - 2 * y[1:-1]) / dx**2
        v[1:-1] += a[1:-1] * dt
        y[1:-1] += v[1:-1] * dt
    return y

# Function to initialize conditions
@njit
def numba_initialize_conditions(nx):
    y = np.exp(-(x - 5)**2)  # Initial displacement
    v = np.zeros(nx)         # Initial velocity
    a = np.zeros(nx)         # Initial acceleration
    return y, v, a

# Optimized time evolution using NumPy and NumBa
@njit(fastmath=True)
def numba_time_evolution(y, v, a, nt, dx, dt):
    for _ in range(nt):
        # Vectorized computation for acceleration
        a[1:-1] = (y[2:] + y[:-2] - 2 * y[1:-1]) / dx**2
        # Update velocity and displacement using vectorized operations
        v[1:-1] += a[1:-1] * dt
        y[1:-1] += v[1:-1] * dt
    return y

# Main Function
def main():
    runs = 5
    total_time = 0.0

    print(f"Running base time evolution {runs} times...")
    # Run time evolution multiple times
    for run in range(runs):
        # Initialize conditions
        y, v, a = initialize_conditions()

        # Measure time for one run of time evolution
        start_time = time.time()
        y_final = time_evolution(y, v, a, nt, dx, dt)
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_time += elapsed_time

        print(f"Run {run + 1}: {elapsed_time:.2f} seconds")

    # Calculate and print average time
    average_time = total_time / runs
    print(f"Final displacement at t = {tmax}: {y_final[256]:.2f}")
    print(f"Average Base Time Evolution Runtime over {runs} runs: {average_time:.2f} seconds")

# Numba optimized
def numba_main():
    runs = 5
    total_time = 0.0

    print(f"Running Numba time evolution {runs} times...")

    # Compile functions by running once (NumBa JIT has compilation overhead)
    y, v, a = numba_initialize_conditions(nx)
    numba_time_evolution(y, v, a, 1, dx, dt)

    for run in range(runs):
        # Initialize conditions
        y, v, a = numba_initialize_conditions(nx)

        # Measure execution time
        start_time = time.time()
        y_final = numba_time_evolution(y, v, a, nt, dx, dt)
        end_time = time.time()

        elapsed_time = end_time - start_time
        total_time += elapsed_time

        print(f"Run {run + 1}: {elapsed_time:.2f} seconds")

    # Calculate average runtime
    average_time = total_time / runs
    print(f"Final displacement at t = {tmax}: {y_final[256]:.2f}")
    print(f"Average Numba Time Evolution Runtime over {runs} runs: {average_time:.2f} seconds")

if __name__ == "__main__":
    main()
    numba_main()
