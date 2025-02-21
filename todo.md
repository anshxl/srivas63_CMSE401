# TODO Checklist: Optimized MLP with Numba

## Phase 1: Environment & Repository Setup
- [X] **Repository Structure**
  - [X] Create project folder (`mlp_numba_project`)
  - [X] Create files: `main.py`, `model.py`, `optimizer.py`, `data_loader.py`, `benchmark.py`, `utils.py`, `low_level_ops.py`, `README.md`
  - [X] Create directories: `/experiments`, `/results`
- [X] **Environment Setup**
  - [X] Set up a Python virtual environment (Python 3.8+)
  - [X] Install required libraries: `numpy`, `numba`, `scipy` (if needed), `matplotlib`, and timing utilities (`time` or `timeit`)
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
  - [X] Hidden Layer 1: Shape `(784, 128)` with ReLU
  - [X] Hidden Layer 2: Shape `(128, 128)` with ReLU
  - [X] Hidden Layer 3: Shape `(128, 128)` with ReLU
  - [X] Output Layer: Shape `(128, 10)` with Softmax
- [X] **Forward Propagation**
  - [X] Implement forward pass for each layer (matrix multiplications, activations)
- [X] **Backward Propagation**
  - [X] Implement gradient computation for each layer
- [X] **Initial Testing**
  - [X] Compare outputs and gradients against a baseline NumPy implementation

## Phase 4: Numba-Based Optimization & Low-Level Ops
- [ ] **Refactor Critical Functions in low_level_ops.py**
  - [ ] Implement custom versions of key NumPy functions (e.g., dot, sum, mean, zeros) using explicit Python loops.
  - [ ] Decorate these functions with `@njit(parallel=True)` to accelerate them.
- [ ] **Integrate Low-Level Ops in the MLP Model**
  - [ ] Refactor forward propagation and gradient computation in `model.py` to optionally use the low-level functions.
  - [ ] Validate that the custom operations produce equivalent results to the NumPy versions.
- [ ] **Benchmarking**
  - [ ] Measure performance improvements of the low-level Numba-accelerated functions vs. the original NumPy implementations.
  - [ ] Update benchmarking scripts in `benchmark.py`.

## Phase 5: Optimizer Integration (Adam & SGD)
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

## Phase 6: Training Loop & Benchmarking
- [X] **Training Loop**
  - [X] Implement the overall training loop in `main.py`
  - [X] Integrate data loading, forward pass, loss calculation, backward pass, and optimizer updates
- [ ] **Batch Processing**
  - [ ] Implement batch handling and experiment with sizes (32, 64, 128, etc.)
- [ ] **Metric Logging**
  - [ ] Log training metrics: loss, accuracy, iteration execution time
- [ ] **Benchmarking Scripts**
  - [ ] Develop scripts in `benchmark.py` to compare NumPy vs. Numba performance

## Phase 7: Testing, Debugging, & Validation
- [ ] **Unit Tests**
  - [ ] Write tests for each activation function (ReLU, Softmax)
  - [ ] Validate forward propagation outputs
  - [ ] Validate backward propagation gradient computations
  - [ ] Test weight update routines
- [ ] **Debugging**
  - [ ] Use `numba.verbose=True` for diagnosing parallelism issues
  - [ ] Address numerical stability issues (e.g., implement log-softmax if needed)
- [ ] **Integration Testing**
  - [ ] Run complete training loop tests to ensure expected behavior

## Phase 8: Documentation & Deployment
- [ ] **Documentation**
  - [ ] Update `README.md` with project overview, setup instructions, and usage guidelines
  - [ ] Add inline code documentation (docstrings) across all modules
- [ ] **Continuous Integration**
  - [ ] (Optional) Set up CI for automated testing
- [ ] **Packaging**
  - [ ] Prepare the project for deployment or sharing

## Phase 9: Stretch Goals (Advanced Optimization)
- [ ] **GPU Acceleration Exploration**
  - [ ] Research and prototype GPU acceleration using `@cuda.jit`
  - [ ] Identify code sections (e.g., matrix multiplications) for GPU optimization
- [ ] **Performance Comparison**
  - [ ] Benchmark GPU-accelerated version against the CPU (Numba) version
- [ ] **Threading & Fast Math**
  - [ ] Experiment with Numba’s threading models (e.g., `fastmath=True`)
- [ ] **Documentation of Findings**
  - [ ] Document performance comparisons and trade-offs

---

# Final Checklist
- [ ] Each phase has been completed and tested
- [ ] All unit and integration tests pass
- [ ] Benchmark results documented and meet performance goals
- [ ] Project documentation is complete and up-to-date
