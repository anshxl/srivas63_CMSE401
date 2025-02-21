# **Optimized MLP with Numba - Developer Specification**

## **Project Overview**
This project focuses on implementing a **fully connected Multi-Layer Perceptron (MLP) from scratch** in Python. The twist is that we will implement two versions:
1. A **baseline version** that uses vectorized NumPy operations.
2. An **optimized version** where critical operations (e.g., matrix multiplication, summation, allocation) are reimplemented with low-level Python loops and then accelerated using **Numba**.

The primary goals are to accelerate **forward propagation, backpropagation, and weight updates** using CPU parallelization (`@njit(parallel=True)`) and to benchmark these low-level implementations against the NumPy-based version. GPU acceleration with `@cuda.jit` is a stretch goal.

## **Project Goals**
- Implement an **efficient, parallelized** MLP for MNIST classification.
- **Benchmark performance** between the baseline NumPy implementation and a custom, low-level Python loop version accelerated with Numba.
- Experiment with different **gradient update strategies** and custom low-level operations to maximize performance.
- Maintain numerical **accuracy consistency** between both implementations.

---
## **1. Requirements**
### **1.1 Technical Requirements**
- **Programming Language:** Python 3.8+
- **Libraries:**
  - `numpy` (for the baseline implementation)
  - `numba` (for JIT compilation and parallelization of custom low-level functions)
  - `scipy` (optional, for numerical stability)
  - `matplotlib` (for visualizing benchmarks & results)
  - `time` or `timeit` (for performance measurement)
- **Hardware:**
  - CPU: Multi-core support required
  - GPU (optional): NVIDIA with CUDA support for future GPU acceleration

### **1.2 Functional Requirements**
1. **Forward Propagation**
   - Baseline: Use vectorized NumPy operations.
   - Optimized: Reimplement critical functions (e.g., dot product, sum, mean, zeros) in Python loops and accelerate them with Numba.
   - Activation functions (ReLU, Softmax) should be optimized for parallel execution.
2. **Backward Propagation**
   - Efficient gradient computation using both baseline and low-level implementations.
   - Optimized weight updates using the Adam optimizer.
3. **Batch Processing**
   - Efficient memory handling to avoid reallocation overhead.
   - Parallelized batch computation in the Numba version.
4. **Performance Benchmarking**
   - Measure execution time per training iteration for both implementations.
   - Compare speedup factors between the NumPy and Numba-accelerated versions.

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
│── main.py                # Entry point for training and evaluation 
│── model.py               # MLP class with forward & backward propagation 
│── optimizer.py           # Adam & SGD implementations 
│── data_loader.py         # MNIST data processing & batching 
│── benchmark.py           # Performance testing and profiling 
│── low_level_ops.py       # Custom implementations of np.dot, np.sum, np.zeros, etc., accelerated with Numba 
│── utils.py               # Helper functions (activation, loss, etc.) 
│── experiments/           # Directory for testing different optimization strategies 
│── results/               # Store benchmarking results 
│── README.md              # Project documentation
```


### **2.3 Optimization Strategies**
- **Matrix multiplications & Element-wise Operations:**  
  - Baseline: Use NumPy’s optimized C routines.
  - Optimized: Reimplement these operations using explicit loops and decorate with `@njit(parallel=True)`.
- **Custom Low-Level Functions:**  
  - Write your own versions of `np.dot`, `np.sum`, `np.mean`, and `np.zeros` to expose Python loop overhead.
  - Accelerate these functions with Numba to compare their performance against the NumPy equivalents.
- **Pre-allocated Arrays:**  
  - Use pre-allocated arrays in the low-level implementation to reduce memory overhead.
- **Parallelization:**  
  - Use Numba’s support for parallel loops (`prange`) to achieve additional speed-ups.

---
## **3. Data Handling**
### **3.1 Dataset**
- **Dataset:** MNIST (28x28 grayscale images, 10 classes)
- **Input Size:** Flattened images `(Batch, 784)`
- **Batch Sizes:** 32, 64, 128 (experiment to find the best performance)

### **3.2 Data Preprocessing**
- **Normalization:** Scale pixel values to `[0, 1]`
- **One-hot Encoding:** Convert labels to one-hot vectors of shape `(Batch, 10)`

---
## **4. Error Handling & Debugging**
### **4.1 Potential Issues & Solutions**
| **Issue**                         | **Solution**                                   |
|------------------------------------|-----------------------------------------------|
| Numerical instability (Softmax)   | Use `log-softmax` or subtract max values to prevent overflow errors  |
| Parallelism issues (Numba)        | Debug with `numba.verbose=True`               |
| Memory allocation bottlenecks     | Use pre-allocated arrays for gradients and intermediate computations        |
| Vanishing gradients               | Use He Initialization and ReLU                |
| Exploding gradients               | Implement gradient clipping (optional)        |

---
## **5. Testing Plan**
### **5.1 Performance Evaluation**
| **Metric**              | **Method**                                              |
|-------------------------|---------------------------------------------------------|
| Execution Time          | Measure time per forward & backward pass using both implementations            |
| Speedup Factor          | Compare NumPy vs. Numba (custom low-level ops) execution speed              |
| Scalability             | Test different batch sizes (32, 64, 128, 256)             |
| Accuracy Consistency    | Ensure no deviation in loss and predictions between versions         |

### **5.2 Unit Testing**
- **Activation Functions:** Ensure correctness of ReLU & Softmax implementations.
- **Forward Propagation:** Check outputs match between baseline and low-level Numba implementations.
- **Backpropagation:** Validate gradient computations.
- **Weight Updates:** Verify that weight updates (Adam/SGD) are correctly applied.

---
## **6. Stretch Goals**
- Implement **GPU acceleration with `@cuda.jit`** for further optimization.
- Experiment with **Numba threading models (`fastmath=True`)** for improved CPU efficiency.
- Compare performance against **PyTorch/TensorFlow implementations**.
