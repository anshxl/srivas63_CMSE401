# **Optimized MLP with GPU Acceleration - Developer Specification**

## **Project Overview**
This project focuses on implementing a **fully connected Multi-Layer Perceptron (MLP) from scratch** in Python. The evolution of our approach is as follows:
1. **Baseline Implementation:**  
   A version built entirely with vectorized NumPy operations.
2. **CPU Optimization with Numba (Failed Attempts):**  
   We experimented with:
   - Rewriting critical operations (forward propagation, backpropagation, and weight updates) using low-level Python loops accelerated with Numba.
   - Using JIT compilation and parallelization (`@njit(parallel=True)`) to speed up batch processing.
   - Implementing our own low-level Python code to replace NumPy operations.
   
   **Outcome:**  
   Despite these efforts, the optimized CPU-based implementations were either equivalent to or slower than the baseline NumPy code because NumPy’s vectorized operations already leverage highly optimized BLAS libraries and multi-threading at the C level.
   
3. **GPU Acceleration with CuPy (Current Direction):**  
   Given the limitations encountered with CPU-based optimizations, we are pivoting to leverage GPU acceleration using CuPy. GPUs are naturally suited to massive parallel computations, particularly for matrix and vector operations central to neural networks.

## **Project Goals**
- Implement an **efficient, parallelized** MLP for MNIST classification.
- Demonstrate the performance of the baseline NumPy implementation.
- Document our initial CPU-based optimization attempts using Numba (including their “failed” performance improvements).
- Develop a GPU-accelerated version using CuPy and benchmark its performance relative to the CPU versions.
- Maintain numerical **accuracy consistency** across all versions.

---

## **1. Requirements**
### **1.1 Technical Requirements**
- **Programming Language:** Python 3.8+
- **Libraries:**
  - `numpy` (for the baseline implementation)
  - `numba` (for our initial CPU-based attempts)
  - `cupy` (for GPU acceleration)
  - `scipy` (optional, for numerical stability)
  - `matplotlib` (for visualizing benchmarks & results)
  - Timing utilities (`time` or `timeit`)
- **Hardware:**
  - **CPU:** Multi-core support with optimized BLAS (e.g., MKL, OpenBLAS)
  - **GPU:** NVIDIA with CUDA support

### **1.2 Functional Requirements**
1. **Forward Propagation**
   - **Baseline:** Use vectorized NumPy operations.
   - **Initial CPU Optimization:** Attempt low-level implementations using Python loops and Numba (`@njit`, `@njit(parallel=True)`).
   - **GPU-Accelerated:** Implement the forward pass using CuPy to fully leverage GPU parallelism.
2. **Backward Propagation**
   - Similar to the forward pass, implement baseline and then experiment with accelerated versions.
3. **Batch Processing & Data Handling**
   - Efficient memory handling and batching (with various sizes).
4. **Performance Benchmarking**
   - Measure execution time per training iteration across:
     - Baseline NumPy version.
     - CPU-optimized Numba versions (documenting the limited benefits).
     - GPU-accelerated CuPy version.
   - Document and compare speedup factors.

---

## **2. Architecture**
### **2.1 Model Architecture**
| Layer           | Shape                |
|-----------------|----------------------|
| Input Layer     | (Batch, 784)         |
| Hidden Layer 1  | (784, 128) + ReLU    |
| Hidden Layer 2  | (128, 128) + ReLU    |
| Hidden Layer 3  | (128, 128) + ReLU    |
| Output Layer    | (128, 10) + Softmax  |

### **2.2 Project Structure**
/mlp_gpu_project 
│── main.py               # Entry point for training and evaluation 
│── model.py              # MLP class with forward & backward propagation 
│── optimizer.py          # Adam & SGD implementations 
│── data_loader.py        # MNIST data processing & batching 
│── benchmark.py          # Performance testing and profiling 
│── low_level_ops.py      # Custom low-level ops for initial Numba experiments 
│── gpu_ops.py            # New module for GPU-accelerated operations with CuPy 
│── utils.py              # Helper functions (activation, loss, etc.) 
│── experiments/          # Directory for testing different optimization strategies 
│── results/              # Store benchmarking results 
│── README.md             # Project documentation


---

## **3. Optimization Strategies**

### **3.1 Baseline & Initial CPU-based Attempts (Failed Approaches)**
- **Vectorized NumPy:**  
  Already optimized via BLAS libraries.
- **Numba JIT and Parallelization:**  
  - Rewriting critical functions (forward, backward, optimizer updates) with low-level Python loops and applying `@njit` and `@njit(parallel=True)`.
  - Implementing batch processing through Numba slowed execution due to overhead and loss of optimized BLAS routines.
- **Low-Level Python Code:**  
  Attempting to manually reimplement operations using built-in Python lists and methods accelerated with Numba did not yield improvements.

### **3.2 GPU Acceleration with CuPy (Current Focus)**
- **CuPy Integration:**  
  - Replace NumPy with CuPy in performance-critical sections.
  - Leverage GPU-accelerated matrix multiplications and vectorized operations.
- **Benchmarking:**  
  Compare GPU execution time with the CPU-based baseline and document improvements.

---

## **4. Data Handling**
### **4.1 Dataset**
- **Dataset:** MNIST (28x28 grayscale images, 10 classes)
- **Input Size:** Flattened images `(Batch, 784)`
- **Batch Sizes:** 32, 64, 128, etc.

### **4.2 Data Preprocessing**
- **Normalization:** Scale pixel values to `[0, 1]`
- **One-hot Encoding:** Convert labels to one-hot vectors

---

## **5. Error Handling & Debugging**
### **5.1 Potential Issues & Solutions**
| **Issue**                         | **Solution**                                   |
|-----------------------------------|-----------------------------------------------|
| Numerical instability (Softmax)   | Use log-softmax or subtract max values        |
| Parallelism issues (Numba/GPU)      | Debug with Numba verbose mode or CuPy logging   |
| Memory allocation bottlenecks      | Use pre-allocated arrays and efficient data transfers |
| Implementation mismatches         | Ensure numerical consistency across versions  |

---

## **6. Testing & Benchmarking**
### **6.1 Performance Evaluation**
| **Metric**              | **Method**                                              |
|-------------------------|---------------------------------------------------------|
| Execution Time          | Compare per-iteration time across NumPy, Numba, and CuPy  |
| Speedup Factor          | Benchmark GPU-accelerated vs. CPU-based implementations   |
| Scalability             | Test various batch sizes and data volumes               |
| Accuracy Consistency    | Validate loss and predictions remain consistent         |

### **6.2 Unit Testing**
- **Activation Functions:** ReLU, Softmax (both CPU and GPU versions)
- **Forward Propagation:** Validate outputs across implementations
- **Backward Propagation:** Verify gradient computations
- **Weight Updates:** Ensure optimizer updates are correct

---

## **7. Stretch Goals**
- Explore additional GPU kernels using CuPy’s raw kernel interface.
- Investigate hybrid models using both CPU and GPU in tandem.
- Compare performance with established frameworks like PyTorch or TensorFlow.

---

# Final Checklist
- [ ] Complete baseline NumPy implementation and verify accuracy.
- [ ] Document initial Numba-based CPU experiments (and their limitations).
- [ ] Develop and integrate GPU-accelerated modules with CuPy.
- [ ] Benchmark and compare performance across all approaches.
- [ ] Update project documentation to reflect the pivot to GPU acceleration.
