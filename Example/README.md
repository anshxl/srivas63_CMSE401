# Numba Optimized MLP Example

## Abstract

This example demonstrates how to accelerate a basic computational loop using Numba. The software evaluated is a programming tool that leverages Python and the Numba JIT compiler to speed up computational routines—common in machine learning, scientific computing, and engineering simulations. By converting Python functions into optimized machine code at runtime, Numba can provide significant speedups for numerical tasks.

## HPCC Installation Instructions

Follow these step-by-step instructions to set up the required environment on the HPCC:

1. **Unload other modules and load Miniforge3:**
   ```bash
   module purge
   module load Miniforge3
   ```

2. **Create a conda environment with the required packages:**
   ```bash
   conda create --name numba_mlp_env python=3.11.5 numpy=1.23.5 numba=0.60.0
   ```

3. **Activate the environment:**
   ```bash
   conda activate numba_mlp_env
   ```

## Example Usage

This directory contains a simple Python script (`example.py`) that demonstrates the performance difference between a pure Python loop and a Numba-optimized loop.

### What the Example Does

- **Pure Python Loop:** Computes the sum of sine values in a loop using standard Python, illustrating a baseline performance.
- **Numba Optimized Loop:** Applies Numba’s `@jit` decorator to the same computation, highlighting the speedup achieved through JIT compilation.

### Running the Example

1. **Ensure you are in the correct conda environment:**
   ```bash
   conda activate numba_mlp_env
   ```

2. **Run the script:**
   ```bash
   python example.py
   ```

The script will execute both versions of the loop and print out the result along with the execution times, allowing you to see the performance benefit of using Numba.

## Files in this Directory

- **README.md:** This file, which includes a description of the software, installation instructions, and usage guidelines.
- **example.py:** A Python script containing both a pure Python and a Numba-optimized loop for demonstration.

## Conclusion

This example provides a quick and practical demonstration of how Numba can accelerate numerical computations in Python. The instructions and code provided here are intended to help future users—such as classmates or fellow researchers—install the software on the HPCC and understand how to leverage Numba for performance-critical tasks.

## References

* An example using 1D wave propagation from HW1, sped up using Numba.
* Installation instructions referenced from ICER's guide to using the HPCC: ICER HPCC Guide