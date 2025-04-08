# This script will run all three models multiple times, and time each run.
# It will also print the average time taken for each model.

import numpy as np
import time
from tqdm import tqdm  # type: ignore
import cupy as cp #type: ignore[import]
from numba import config #type: ignore[import]

from adam import AdamOptimizer # Used for baseline and numba models
from cupy_optimization.gpu_adam import GPUAdamOptimizer # Used for GPU model

from baseline.model_baseline import NeuralNetwork as BaselineNN 
from numba_optimization.numba_model import NeuralNetwork as NumbaNN
from cupy_optimization.gpu_model import GPUNeuralNetwork as GPU_NN

from data_loader import load_data

# Load data
(X_train, y_train), (X_test, y_test) = load_data()

# Convert data to GPU arrays for GPU model
X_train_gpu = cp.asarray(X_train)
y_train_gpu = cp.asarray(y_train)
X_test_gpu = cp.asarray(X_test)
y_test_gpu = cp.asarray(y_test)

# Train and evaluate the Baseline model 10 times, and time each run
def benchmark_baseline():
    print("Benchmarking Baseline Model...")
    baseline_model = BaselineNN(784, 128, 10)
    runtimes = []
    for i in tqdm(range(5), desc="Baseline Model"):
        # Train the model and time the training process
        start_time = time.time()
        baseline_model.train(X_train, y_train, learning_rate=0.001, epochs=40, verbose=False)
        end_time = time.time()

        # Print the training time
        # print(f"Training time (Baseline): {end_time - start_time:.2f} seconds")
        runtimes.append(end_time - start_time)

    # Print the average training time
    avg_runtime = np.mean(runtimes)
    print(f"Average training time (Baseline): {avg_runtime:.2f} seconds")

    # Evaluate the model
    accuracy = baseline_model.evaluate(X_test, y_test)
    print(f"Test accuracy (Baseline): {accuracy:.2f}%")

# Train and evaluate the Numba model 10 times, and time each run
def benchmark_numba():
    print("Benchmarking Numba Model...")
    numba_model = NumbaNN(784, 128, 10)
    runtimes = []
    for i in tqdm(range(5), desc="Numba Model"):

        # Warm up the Numba model
        print("Warming up...")
        numba_model.warmup()
        # Train the model and time the training process
        start_time = time.time()
        numba_model.train(X_train, y_train, learning_rate=0.001, epochs=40, verbose=False)
        end_time = time.time()

        # Print the training time
        # print(f"Training time (Numba): {end_time - start_time:.2f} seconds")
        runtimes.append(end_time - start_time)

    # Print the average training time
    avg_runtime = np.mean(runtimes)
    print(f"Average training time (Numba): {avg_runtime:.2f} seconds")

    # Evaluate the model
    accuracy = numba_model.evaluate(X_test, y_test)
    print(f"Test accuracy (Numba): {accuracy:.2f}%")

# Train and evaluate the GPU model 10 times, and time each run
def benchmark_gpu():
    print("Benchmarking GPU Model...")
    gpu_model = GPU_NN(X_train_gpu.shape[1], 128, y_train_gpu.shape[1])
    runtimes = []
    for i in tqdm(range(5), desc="GPU Model"):

        # Train the model and time the training process
        start_time = time.time()
        gpu_model.train(X_train_gpu, y_train_gpu, learning_rate=0.001, epochs=40, verbose=False)
        end_time = time.time()

        # Print the training time
        # print(f"Training time (GPU): {end_time - start_time:.2f} seconds")
        runtimes.append(end_time - start_time)

    # Print the average training time
    avg_runtime = np.mean(runtimes)
    print(f"Average training time (GPU): {avg_runtime:.2f} seconds")

    # Evaluate the model
    accuracy = gpu_model.evaluate(X_test_gpu, y_test_gpu)
    print(f"Test accuracy (GPU): {accuracy:.2f}%")

if __name__ == "__main__":
    # Set Numba to use the CPU for this benchmark
    config.THREADING_LAYER = 'workqueue'
    print("Numba threading layer set to workqueue for CPU execution.")
    print("NUMBA_DEFAULT_NUM_THREADS:", config.NUMBA_DEFAULT_NUM_THREADS)

    # Run the benchmarks
    benchmark_baseline()
    benchmark_numba()
    benchmark_gpu()
