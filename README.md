# Optimized MLP with GPU Acceleration

This repository contains an implementation of a multi-layer perceptron (MLP) designed to explore performance optimizations on both CPU and GPU. The project starts from a baseline NumPy implementation and includes experimental CPU optimization attempts via Numba, as well as GPU acceleration using CuPy.

The goal is to evaluate different optimization strategies, compare their performance, and provide detailed benchmarks. Although the CPU-based experiments demonstrated that some approaches yield no significant improvement over the highly optimized NumPy routines, GPU acceleration has shown promise in speeding up key operations.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Data Handling & Preprocessing](#data-handling--preprocessing)
- [Core MLP Model Development](#core-mlp-model-development)
- [CPU Optimization Experiments](#cpu-optimization-experiments)
- [GPU Optimization with CuPy](#gpu-optimization-with-cupy)
- [Optimizer Integration](#optimizer-integration)
- [Training, Benchmarking & Testing](#training-benchmarking--testing)
- [Documentation & Deployment](#documentation--deployment)
- [Stretch Goals & Future Work](#stretch-goals--future-work)
- [License](#license)

---

## Overview

This project investigates different optimization strategies for training an MLP model by:
- Implementing a baseline MLP using NumPy.
- Attempting CPU optimizations using two Numba-based approaches.
- Developing GPU-accelerated implementations with CuPy.
- Integrating common optimization algorithms (Adam and SGD).
- Benchmarking the performance across the different implementations.

The experiments include:
1. **Low-Level Python Implementation for Numba:** Developing a low-level Python version of the MLP to later enable Numba optimization. This approach was abandoned because Numba optimization requires input data to be stored in NumPy arrays.
2. **Numba-Decorated NumPy Implementation:** Decorating the core functions of the NumPy-based MLP with `@njit` to enable parallelism via Numba. However, this approach did not yield a measurable performance gain since NumPy is already highly optimized.
3. **GPU Acceleration with CuPy:** Using CuPy to create GPU-accelerated versions of critical operations to significantly improve performance over CPU-based implementations.

---

## Repository Structure

```
mlp_gpu_project/
├── main.py               # Main training loop and integration of modules
├── model.py              # Contains the MLP class and model architecture
├── optimizer.py          # Defines the optimizer interface and implementations (Adam & SGD)
├── data_loader.py        # Module to download and load the MNIST dataset
├── benchmark.py          # Benchmarking scripts for comparing implementations
├── utils.py              # Utility functions and helper methods
├── low_level_ops.py      # Contains CPU-based optimization experiments using low-level Python code and Numba
├── gpu_ops.py            # Contains GPU-accelerated operations implemented with CuPy
├── README.md             # This README file
└── /experiments          # Folder for experimental scripts and analysis
└── /results              # Folder to store benchmark results and performance logs
```

---

## Environment Setup

The project requires Python 3.8 or higher. A virtual environment is recommended to isolate dependencies. To get started:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/mlp_gpu_project.git
   cd mlp_gpu_project
   ```

2. **Set Up a Virtual Environment:**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install Required Libraries:**

   ```bash
   pip install numpy numba cupy scipy matplotlib tqdm tensorflow
   ```

4. **HPCC Instructios:**
  These can be found under the `Example` directory, in the README.md file

---

## Data Handling & Preprocessing

The data handling components are designed to work with the MNIST dataset.

- **Data Loader:**
  - Implemented in `data_loader.py`.
  - Downloads and loads the MNIST dataset.
- **Preprocessing Steps:**
  - Normalizes pixel values to the range [0, 1].
  - Flattens each 28×28 image into a 784-dimensional vector.
  - Applies one-hot encoding to the labels for 10 classes.
- **Unit Testing:**
  - Unit tests are included to verify data loading, normalization, and one-hot encoding.

---

## Core MLP Model Development

- **Model Definition:**
  - The MLP class is defined in `model_baseline.py`.
  - The model architecture includes:
    - **Input Layer:** Accepts batches of 784-dimensional vectors.
    - **Hidden Layers:** Employ ReLU activations.
    - **Output Layer:** Uses Softmax for classification.
- **Forward & Backward Propagation:**
  - Implemented using standard NumPy operations.
  - The forward pass calculates activations for each layer.
  - Backpropagation computes gradients for weight updates.
- **Initial Testing:**
  - Early experiments compared outputs and gradients against expected results to ensure correctness.

---

## CPU Optimization Experiments

### Low-Level Python Implementation for Numba
- A low-level Python version of the MLP was implemented to serve as a base for Numba optimization.
- This approach ultimately failed because Numba requires data to be provided as NumPy arrays, limiting its applicability.

### Numba-Decorated NumPy Implementation
- The core functions in the existing NumPy implementation were decorated with `@njit` to leverage Numba's speed-up.
- This MLP class is defined in `numba_model.py`
- Benchmarks revealed no significant performance improvement because the underlying NumPy operations are already highly optimized.

---

## GPU Optimization with CuPy

### GPU Operations Module
- **gpu_ops.py:** A new module designed to implement GPU-accelerated versions of critical operations such as:
  - Matrix multiplications.
  - Activation functions (e.g., ReLU, Softmax).
- **CuPy Integration:**
  - The MLP class or a subclass has been modified to operate on CuPy arrays.
  - GPU versions of forward propagation and weight update steps have been implemented.

### Advanced GPU Kernel Optimization (Optional)
- Exploration of raw CuPy kernels or other advanced features for further acceleration is planned.

### Benchmarking
- Benchmarking scripts in `benchmark.py` are set up to compare:
  - Baseline NumPy implementations.
  - CPU-based Numba optimizations.
  - GPU-accelerated implementations using CuPy.

---

## Optimizer Integration

- **Interface Definition:**
  - A common interface for optimizers is defined in `adam.py`.
- **Implemented Optimizers:**
  - **Adam Optimizer:** Provides adaptive learning rate updates with standard hyperparameters.
  - **SGD Optimizer:** A fallback option for standard stochastic gradient descent.
- **Integration & Testing:**
  - The optimizers are integrated into the MLP class.
  - Unit tests ensure that weight updates are correctly applied.

---

## Training, Benchmarking & Testing

### Training Loop
- The overall training loop is implemented in `training_baseline.py` and includes:
  - Data loading.
  - Forward pass.
  - Loss computation.
  - Backward pass.
  - Weight updates via the chosen optimizer.

### Benchmarking & Batch Processing
- Batch handling and metric logging (loss, accuracy, and execution time) are part of the training workflow.
- Benchmarking scripts in `benchmark.py` compare the performance of:
  - Vectorized NumPy computations.
  - Numba-enhanced CPU implementations.
  - CuPy-based GPU implementations.

### Testing, Debugging & Validation
- **Unit Tests:**
  - Include tests for activation functions (ReLU, Softmax), forward propagation, gradient correctness, and optimizer updates.
- **Debugging Tools:**
  - Use `numba.verbose=True` to troubleshoot CPU parallelism.
  - Implement logging for CuPy operations.

---

## Documentation & Deployment

### Project Documentation
- Inline code documentation and comprehensive docstrings are present across all modules.
- This README provides an overview, but further detailed documentation is maintained within the code and supplemental files.

### Continuous Integration & Packaging
- Optional CI pipelines (e.g., GitHub Actions) can be set up for automated testing.
- The repository is structured to facilitate packaging and sharing for deployment in various environments.

---

## Stretch Goals & Future Work

- **Additional GPU Optimizations:**
  - Explore the use of raw CuPy kernels or Numba’s `@cuda.jit` for further performance improvements.
- **Hybrid CPU-GPU Models:**
  - Investigate strategies for concurrent utilization of CPU and GPU to maximize throughput.
- **Performance Documentation:**
  - Detailed performance benchmarks and trade-off analyses will be documented to guide future optimizations and research.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This project builds upon extensive work in the Python scientific computing community, especially the contributions from the NumPy, Numba, and CuPy projects.

---

*For any questions or contributions, please feel free to open an issue or create a pull request on GitHub.*
