FJFormer 🔮
==========
FJFormer is a powerful and flexible JAX-based package designed to accelerate and simplify machine learning and deep learning workflows. It provides a comprehensive suite of tools and utilities for efficient model development, training, and deployment.

Features
----------

1. JAX Sharding Utils
Leverage the power of distributed computing and model parallelism with our advanced JAX sharding utilities. These tools enable efficient splitting and management of large models across multiple devices, enhancing performance and enabling the training of larger models.

2. Custom Pallas / Triton Operation Kernels
Boost your model's performance with our optimized kernels for specific operations. These custom-built kernels, implemented using Pallas and Triton, provide significant speedups for common bottleneck operations in deep learning models.

3. Pre-built Optimizers
Jump-start your training with our collection of ready-to-use, efficiently implemented optimization algorithms:
- **AdamW**: An Adam variant with decoupled weight decay.
- **Adafactor**: Memory-efficient adaptive optimization algorithm.
- **Lion**: Recently proposed optimizer combining the benefits of momentum and adaptive methods.
- **RMSprop**: Adaptive learning rate optimization algorithm.

4. Utility Functions
A rich set of utility functions to streamline your workflow, including:
- Various loss functions (e.g., cross-entropy)
- Metrics calculation
- Data preprocessing tools

5. ImplicitArray
Our innovative ImplicitArray class provides a powerful abstraction for representing and manipulating large arrays without instantiation. Benefits include:
- Lazy evaluation for memory efficiency
- Optimized array operations in JAX
- Seamless integration with other FJFormer components

6. Custom Dtypes

- Implement 4-bit quantization (NF4) effortlessly using our ArrayNF4 class, built on top of ImplicitArray. Reduce model size and increase inference speed without significant loss in accuracy.

- Similar to ArrayNF4, our Array8Lt implementation offers 8-bit quantization via ImplicitArray, providing a balance between model compression and precision.

7. LoRA (Low-Rank Adaptation)
Efficiently fine-tune large language models with our LoRA implementation, leveraging ImplicitArray for optimal performance and memory usage.

8. JAX and Array Manipulation
A comprehensive set of tools and utilities for efficient array operations and manipulations in JAX, designed to complement and extend JAX's native capabilities.

9. Checkpoint Managers
Robust utilities for managing model checkpoints, including:
- Efficient saving and loading of model states
- Version control for checkpoints
- Integration with distributed training workflows
.. _FJFormer:

Zare Chavoshi, Erfan. "FJFormer is a collection of functions and utilities that can help with various tasks when using Flax and JAX.""


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: APIs

    api_docs/APIs


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Getting Started

    contributing