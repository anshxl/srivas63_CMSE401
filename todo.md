# TODO Checklist: Optimized MLP with Numba

## Phase 1: Environment & Repository Setup
- [ ] **Repository Structure**
  - [ ] Create project folder (`mlp_numba_project`)
  - [ ] Create files: `main.py`, `model.py`, `optimizer.py`, `data_loader.py`, `benchmark.py`, `utils.py`, `README.md`
  - [ ] Create directories: `/experiments`, `/results`
- [ ] **Environment Setup**
  - [ ] Set up a Python virtual environment (Python 3.8+)
  - [ ] Install required libraries: `numpy`, `numba`, `scipy` (if needed), `matplotlib`, and timing utilities (`time` or `timeit`)
- [ ] **Version Control**
  - [ ] Initialize a Git repository
  - [ ] Document setup steps in `README.md`

## Phase 2: Data Handling & Preprocessing
- [ ] **Data Loader**
  - [ ] Write a function in `data_loader.py` to download/load the MNIST dataset
- [ ] **Data Preprocessing**
  - [ ] Normalize pixel values to the range [0, 1]
  - [ ] Flatten images from 28×28 to a 784-dimensional vector
  - [ ] Implement one-hot encoding for labels (10 classes)
- [ ] **Unit Testing**
  - [ ] Create tests to verify data loading accuracy
  - [ ] Create tests to confirm correct normalization and one-hot encoding

## Phase 3: Core MLP Model Development
- [ ] **MLP Class Skeleton**
  - [ ] Define an MLP class in `model.py`
- [ ] **Model Architecture Implementation**
  - [ ] Input Layer: Shape `(Batch, 784)`
  - [ ] Hidden Layer 1: Shape `(784, 128)` with ReLU
  - [ ] Hidden Layer 2: Shape `(128, 128)` with ReLU
  - [ ] Hidden Layer 3: Shape `(128, 128)` with ReLU
  - [ ] Output Layer: Shape `(128, 10)` with Softmax
- [ ] **Forward Propagation**
  - [ ] Implement forward pass for each layer (matrix multiplications, activations)
- [ ] **Backward Propagation**
  - [ ] Implement gradient computation for each layer
- [ ] **Initial Testing**
  - [ ] Compare outputs and gradients against a baseline NumPy implementation

## Phase 4: Numba-Based Optimization
- [ ] **Refactoring Forward Propagation**
  - [ ] Decorate critical functions with `@njit(parallel=True)`
  - [ ] Replace Python loops with `numba.prange` where applicable
- [ ] **Refactoring Backward Propagation**
  - [ ] Apply similar Numba optimizations to gradient calculations
- [ ] **Validation**
  - [ ] Verify numerical accuracy of optimized functions against the baseline
- [ ] **Benchmarking**
  - [ ] Measure performance improvements over the pure NumPy version

## Phase 5: Optimizer Integration (Adam & SGD)
- [ ] **Optimizer Interface**
  - [ ] Define a common interface for optimizers in `optimizer.py`
- [ ] **Adam Optimizer**
  - [ ] Implement Adam update rules with appropriate hyperparameters
- [ ] **SGD Optimizer**
  - [ ] Implement a fallback SGD optimizer
- [ ] **Integration**
  - [ ] Integrate optimizer functionality into the MLP class
- [ ] **Testing**
  - [ ] Write unit tests to verify correct weight updates

## Phase 6: Training Loop & Benchmarking
- [ ] **Training Loop**
  - [ ] Implement the overall training loop in `main.py`
  - [ ] Integrate data loading, forward pass, loss calculation, backward pass, and optimizer updates
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

