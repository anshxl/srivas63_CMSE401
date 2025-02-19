# **Optimized MLP with Numba - Developer Specification**

## **Project Overview**
This project focuses on implementing a **fully connected Multi-Layer Perceptron (MLP) from scratch** in Python using **NumPy and Numba** for performance optimization. The primary goal is to accelerate **forward propagation, backpropagation, and weight updates** using CPU parallelization (`@njit(parallel=True)`). GPU parallelization with `@cuda.jit` will be considered as a stretch goal.

## **Project Goals**
- Implement an **efficient, parallelized** MLP for MNIST classification.
- Benchmark performance against a pure **NumPy-based implementation**.
- Experiment with different **gradient update strategies** to find the most efficient.
- Maintain numerical **accuracy consistency** between optimized and baseline versions.

---
## **1. Requirements**
### **1.1 Technical Requirements**
- **Programming Language:** Python 3.8+
- **Libraries:**
  - `numpy` (for tensor operations)
  - `numba` (for JIT compilation and parallelization)
  - `scipy` (optional, for numerical stability)
  - `matplotlib` (for visualizing benchmarks & results)
  - `time` or `timeit` (for performance measurement)
- **Hardware:**
  - CPU: Multi-core support required
  - GPU (optional): NVIDIA with CUDA support for future GPU acceleration

### **1.2 Functional Requirements**
1. **Forward Propagation**
   - Matrix multiplications using `@njit(parallel=True)`
   - Activation functions (ReLU, Softmax) optimized for parallel execution
2. **Backward Propagation**
   - Efficient gradient computation
   - Optimized weight updates using Adam optimizer (fallback to SGD if needed)
3. **Batch Processing**
   - Efficient memory handling to avoid reallocation overhead
   - Parallelized batch computation
4. **Performance Benchmarking**
   - Measure execution time per training iteration
   - Compare speedup vs. NumPy implementation

---
## **2. Architecture**
### **2.1 Model Architecture**
| Layer           | Shape                |
|----------------|----------------------|
| Input Layer    | (Batch, 784)         |
| Hidden Layer 1 | (784, 128) + ReLU    |
| Hidden Layer 2 | (128, 128) + ReLU    |
| Hidden Layer 3 | (128, 128) + ReLU    |
| Output Layer   | (128, 10) + Softmax  |

### **2.2 Project Structure**
```
/mlp_numba_project
│── main.py                   # Entry point for training and evaluation
│── model.py                  # MLP class with forward & backward propagation
│── optimizer.py               # Adam & SGD implementations
│── data_loader.py             # MNIST data processing & batching
│── benchmark.py               # Performance testing and profiling
│── utils.py                   # Helper functions (activation, loss, etc.)
│── experiments/               # Directory for testing different optimization strategies
│── results/                   # Store benchmarking results
│── README.md                  # Project documentation
```

### **2.3 Optimization Strategies**
- **Matrix multiplications:** `@njit(parallel=True)` for parallelized execution
- **Element-wise operations:** Use Numba’s `prange` for parallelization
- **Pre-allocated arrays:** Reduce memory overhead by avoiding frequent allocation

---
## **3. Data Handling**
### **3.1 Dataset**
- **Dataset:** MNIST (28x28 grayscale images, 10 classes)
- **Input Size:** Flattened images `(Batch, 784)`
- **Batch Sizes:** 32, 64, 128 (experiment for best performance)

### **3.2 Data Preprocessing**
- **Normalization:** Scale pixel values to `[0, 1]`
- **One-hot Encoding:** Convert labels to one-hot vectors of shape `(Batch, 10)`

---
## **4. Error Handling & Debugging**
### **4.1 Potential Issues & Solutions**
| **Issue**                         | **Solution**                                   |
|------------------------------------|-----------------------------------------------|
| Numerical instability (Softmax)   | Use `log-softmax` to prevent overflow errors  |
| Parallelism issues (Numba)        | Debug with `numba.verbose=True`               |
| Memory allocation bottlenecks     | Use pre-allocated arrays for gradients        |
| Vanishing gradients               | Use He Initialization and ReLU                |
| Exploding gradients               | Implement gradient clipping (optional)        |

---
## **5. Testing Plan**
### **5.1 Performance Evaluation**
| **Metric**              | **Method**                                              |
|-------------------------|------------------------------------------------------|
| Execution Time          | Measure time per forward & backward pass            |
| Speedup Factor         | Compare NumPy vs. Numba execution speed              |
| Scalability            | Test different batch sizes (32, 64, 128, 256)        |
| Accuracy Consistency   | Ensure no deviation in loss and predictions         |

### **5.2 Unit Testing**
- **Activation Functions:** Ensure correctness of ReLU & Softmax
- **Forward Propagation:** Check outputs match NumPy implementation
- **Backpropagation:** Validate gradient calculations
- **Weight Updates:** Ensure proper weight changes using Adam/SGD

---
## **6. Stretch Goals**
- Implement **GPU acceleration with `@cuda.jit`** for further optimization
- Experiment with **Numba threading models (`fastmath=True`)** for improved CPU efficiency
- Compare performance against **PyTorch/TensorFlow implementations**
