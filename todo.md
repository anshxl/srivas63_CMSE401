# TODO Checklist: Optimized MLP with GPU Acceleration

## Phase 1: Environment & Repository Setup
- [X] **Repository Structure**
  - [X] Create project folder (`mlp_gpu_project`)
  - [X] Create files: `main.py`, `model.py`, `optimizer.py`, `data_loader.py`, `benchmark.py`, `utils.py`, `low_level_ops.py`, `gpu_ops.py`, `README.md`
  - [X] Create directories: `/experiments`, `/results`
- [X] **Environment Setup**
  - [X] Set up a Python virtual environment (Python 3.8+)
  - [X] Install required libraries: `numpy`, `numba`, `cupy`, `scipy` (if needed), `matplotlib`, and timing utilities (`time` or `timeit`)
- [X] **Version Control**
  - [X] Initialize a Git repository
  - [X] Document setup steps in `README.md`

## Phase 2: Data Handling & Preprocessing
- [X] **Data Loader**
  - [X] Write a function in `data_loader.py` to download/load the MNIST dataset
- [X] **Data Preprocessing**
  - [X] Normalize pixel values to the range [0, 1]
  - [X] Flatten images from 28×28 to a 784-dimensional vector
  - [X] Implement one-hot encoding for labels (10 classes)
- [X] **Unit Testing**
  - [X] Create tests to verify data loading accuracy
  - [X] Create tests to confirm correct normalization and one-hot encoding

## Phase 3: Core MLP Model Development
- [X] **MLP Class Skeleton**
  - [X] Define an MLP class in `model.py`
- [X] **Model Architecture Implementation**
  - [X] Input Layer: Shape `(Batch, 784)`
  - [X] Hidden Layers & Output Layer with respective activations (ReLU, Softmax)
- [X] **Forward Propagation**
  - [X] Implement forward pass for each layer (using NumPy)
- [X] **Backward Propagation**
  - [X] Implement gradient computation for each layer
- [X] **Initial Testing**
  - [X] Compare outputs and gradients against expected results

## Phase 4: CPU-based Optimization Experiments (Failed Approaches)
- [X] **Refactor Critical Functions in low_level_ops.py**
  - [X] Implement custom versions of key NumPy functions (e.g., dot, sum, mean, zeros) using explicit Python loops.
  - [X] Decorate these functions with `@njit(parallel=True)` to accelerate them.
- [X] **Integrate Low-Level Ops in the MLP Model**
  - [X] Optionally refactor forward and gradient computations to use these low-level functions.
  - [ ] Benchmark and document that these approaches did not outperform the vectorized NumPy baseline.

## Phase 5: GPU Acceleration Implementation with CuPy
- [ ] **GPU Operations Module**
  - [ ] Create `gpu_ops.py` with GPU-accelerated equivalents for critical operations (e.g., matrix multiplication, activation functions).
- [ ] **Integrate CuPy into the MLP Model**
  - [ ] Modify the MLP class (or create a subclass) to use CuPy arrays and operations for forward propagation and weight updates.
- [ ] **Benchmarking**
  - [ ] Measure performance improvements of the GPU version against both the baseline NumPy and the CPU-based Numba versions.
  - [ ] Update benchmarking scripts in `benchmark.py`.

## Phase 6: Optimizer Integration (Adam & SGD)
- [X] **Optimizer Interface**
  - [X] Define a common interface for optimizers in `optimizer.py`
- [X] **Adam Optimizer**
  - [X] Implement Adam update rules with appropriate hyperparameters
- [X] **SGD Optimizer** (Fallback)
  - [X] Implement a fallback SGD optimizer
- [X] **Integration**
  - [X] Integrate optimizer functionality into the MLP class
- [X] **Testing**
  - [X] Write unit tests to verify correct weight updates

## Phase 7: Training Loop & Benchmarking
- [X] **Training Loop**
  - [X] Implement the overall training loop in `main.py`
  - [X] Integrate data loading, forward pass, loss calculation, backward pass, and optimizer updates
- [ ] **Batch Processing**
  - [ ] Implement and experiment with batch handling and various sizes
- [ ] **Metric Logging**
  - [ ] Log training metrics: loss, accuracy, execution time per iteration
- [ ] **Benchmarking Scripts**
  - [ ] Develop scripts in `benchmark.py` to compare NumPy vs. Numba vs. CuPy performance

## Phase 8: Testing, Debugging, & Validation
- [ ] **Unit Tests**
  - [ ] Write tests for activation functions (ReLU, Softmax) for both CPU and GPU versions
  - [ ] Validate forward propagation outputs and gradients
  - [ ] Test optimizer weight updates
- [ ] **Debugging**
  - [ ] Use `numba.verbose=True` for diagnosing CPU parallelism issues and appropriate logging for CuPy
- [ ] **Integration Testing**
  - [ ] Run complete training loop tests to ensure expected behavior

## Phase 9: Documentation & Deployment
- [ ] **Documentation**
  - [ ] Update `README.md` with project overview, setup instructions, and usage guidelines
  - [ ] Add inline code documentation (docstrings) across all modules
- [ ] **Continuous Integration**
  - [ ] (Optional) Set up CI for automated testing
- [ ] **Packaging**
  - [ ] Prepare the project for deployment or sharing

## Phase 10: Stretch Goals (Advanced Optimization & Analysis)
- [ ] **Additional GPU Optimizations**
  - [ ] Explore using raw CuPy kernels or Numba’s `@cuda.jit` for further acceleration.
- [ ] **Hybrid CPU-GPU Models**
  - [ ] Investigate strategies for concurrent CPU and GPU utilization.
- [ ] **Performance Documentation**
  - [ ] Document and analyze all performance benchmarks and trade-offs across different implementations.

---

# Final Checklist
- [ ] All phases have been completed and tested.
- [ ] Unit and integration tests pass across all implementations.
- [ ] Benchmark results are documented and meet performance expectations.
- [ ] Project documentation is complete and up-to-date.
