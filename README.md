# FJFormer

[![PyPI version](https://badge.fury.io/py/fjformer.svg)](https://badge.fury.io/py/fjformer)
[![Documentation Status](https://readthedocs.org/projects/fjformer/badge/?version=latest)](https://fjformer.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

FJFormer is a powerful and flexible JAX-based package designed to accelerate and simplify machine learning and deep learning workflows. It provides a comprehensive suite of tools and utilities for efficient model development, training, and deployment.

## Features

### 1. JAX Sharding Utils
Leverage the power of distributed computing and model parallelism with our advanced JAX sharding utilities. These tools enable efficient splitting and management of large models across multiple devices, enhancing performance and enabling the training of larger models.

### 2. Custom Pallas / Triton Operation Kernels
Boost your model's performance with our optimized kernels for specific operations. These custom-built kernels, implemented using Pallas and Triton, provide significant speedups for common bottleneck operations in deep learning models.

### 3. Pre-built Optimizers
Jump-start your training with our collection of ready-to-use, efficiently implemented optimization algorithms:
- **AdamW**: An Adam variant with decoupled weight decay.
- **Adafactor**: Memory-efficient adaptive optimization algorithm.
- **Lion**: Recently proposed optimizer combining the benefits of momentum and adaptive methods.
- **RMSprop**: Adaptive learning rate optimization algorithm.

### 4. Utility Functions
A rich set of utility functions to streamline your workflow, including:
- Various loss functions (e.g., cross-entropy)
- Metrics calculation
- Data preprocessing tools

### 5. ImplicitArray
Our innovative ImplicitArray class provides a powerful abstraction for representing and manipulating large arrays without instantiation. Benefits include:
- Lazy evaluation for memory efficiency
- Optimized array operations in JAX
- Seamless integration with other FJFormer components

### 6. Custom Dtypes

- Implement 4-bit quantization (NF4) effortlessly using our Array4Bit class, built on top of ImplicitArray. Reduce model size and increase inference speed without significant loss in accuracy.

- Similar to Array4Bit, our Array4Lt implementation offers 8-bit quantization via ImplicitArray, providing a balance between model compression and precision.

### 7. LoRA (Low-Rank Adaptation)
Efficiently fine-tune large language models with our LoRA implementation, leveraging ImplicitArray for optimal performance and memory usage.

### 8. JAX and Array Manipulation
A comprehensive set of tools and utilities for efficient array operations and manipulations in JAX, designed to complement and extend JAX's native capabilities.

### 9. Checkpoint Managers
Robust utilities for managing model checkpoints, including:
- Efficient saving and loading of model states
- Version control for checkpoints
- Integration with distributed training workflows

## Installation

You can install FJFormer using pip:

```bash
pip install fjformer
```

For the latest development version, you can install directly from GitHub:

```bash
pip install git+https://github.com/yourusername/fjformer.git
```

## Documentation

For detailed documentation, including API references, please visit:

[https://fjformer.readthedocs.org](https://fjformer.readthedocs.org)

## License

FJFormer is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
