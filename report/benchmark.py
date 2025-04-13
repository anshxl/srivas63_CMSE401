# This script will run all three models multiple times, and time each run.
# It will also print the average time taken for each model.

import numpy as np
import time
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress TensorFlow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to hide warnings too

from tqdm import tqdm  # type: ignore
import cupy as cp #type: ignore[import]
from numba import config #type: ignore[import]

from adam import AdamOptimizer # Used for baseline and numba models
from cupy_optimization.gpu_adam import GPUAdamOptimizer # Used for GPU model

from baseline.model_baseline import NeuralNetwork as BaselineNN 
from numba_optimization.numba_model import NeuralNetwork as NumbaNN
from cupy_optimization.gpu_model import GPUNeuralNetwork as GPU_NN

from data_loader import load_data

# Set the random seed for reproducibility
np.random.seed(42)

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
    runtimes = []
    accuracies = []
    for i in tqdm(range(5), desc="Baseline Model"):
        baseline_model = BaselineNN(784, 128, 10)
        # Train the model and time the training process
        start_time = time.time()
        baseline_model.train(X_train, y_train, learning_rate=0.001, epochs=40, verbose=False)
        end_time = time.time()

        # Print the training time
        # print(f"Training time (Baseline): {end_time - start_time:.2f} seconds")
        runtimes.append(end_time - start_time)

        # Evaluate the model
        accuracy = baseline_model.evaluate(X_test, y_test)
        accuracies.append(accuracy)

    # Print the average training time
    avg_runtime = np.mean(runtimes)
    print(f"Average training time (Baseline): {avg_runtime:.2f} seconds")

    # Print the average accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"Average test accuracy (Baseline): {avg_accuracy:.2f}%\n")

    # Plot the loss values
    baseline_model.plot_loss()

# Train and evaluate the Numba model 10 times, and time each run
def benchmark_numba():
    print("Benchmarking Numba Model...")
    runtimes = []
    accuracies = []
    for i in tqdm(range(5), desc="Numba Model"):
        numba_model = NumbaNN(784, 128, 10)
        # Warm up the Numba model
        numba_model.warmup()
        # Train the model and time the training process
        start_time = time.time()
        numba_model.train(X_train, y_train, learning_rate=0.001, epochs=40, verbose=False)
        end_time = time.time()

        # Print the training time
        # print(f"Training time (Numba): {end_time - start_time:.2f} seconds")
        runtimes.append(end_time - start_time)

        # Evaluate the model
        accuracy = numba_model.evaluate(X_test, y_test)
        accuracies.append(accuracy)

    # Print the average training time
    avg_runtime = np.mean(runtimes)
    print(f"Average training time (Numba): {avg_runtime:.2f} seconds")

    # Print the average accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"Average test accuracy (Numba): {avg_accuracy:.2f}%\n")

    # Plot the loss values
    numba_model.plot_loss()

# Train and evaluate the GPU model 10 times, and time each run
def benchmark_gpu():
    print("Benchmarking GPU Model...")
    runtimes = []
    accuracies = []
    for i in tqdm(range(5), desc="GPU Model"):
        gpu_model = GPU_NN(X_train_gpu.shape[1], 128, y_train_gpu.shape[1])
        # Train the model and time the training process
        start_time = time.time()
        gpu_model.train(X_train_gpu, y_train_gpu, learning_rate=0.001, epochs=40, verbose=False)
        end_time = time.time()


        runtimes.append(end_time - start_time)

        # Evaluate the model
        accuracy = gpu_model.evaluate(X_test_gpu, y_test_gpu)
        accuracies.append(accuracy.get())

    # Print the average training time
    avg_runtime = np.mean(runtimes)
    print(f"Average training time (GPU): {avg_runtime:.2f} seconds")

    # Print the average accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"Average test accuracy (GPU): {avg_accuracy:.2f}%\n")

    # Plot the loss values
    gpu_model.plot_loss()

if __name__ == "__main__":
    # Set Numba to use the CPU for this benchmark
    config.THREADING_LAYER = 'workqueue'
    # print("Numba threading layer set to workqueue for CPU execution.")
    max_threads = os.cpu_count()
    config.NUMBA_NUM_THREADS = max_threads
    print("Configured maximum threads:", config.NUMBA_NUM_THREADS)
    print()

    # Run the benchmarks
    benchmark_baseline()
    benchmark_numba()
    benchmark_gpu()
