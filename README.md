# eformer (EasyDel Former)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-Compatible-brightgreen)](https://github.com/google/jax)

**eformer** (EasyDel Former) is a utility library designed to simplify and enhance the development of machine learning models using JAX. It provides a collection of tools for sharding, custom PyTrees, quantization, and optimized operations, making it easier to build and scale models efficiently.

## Features

- **Sharding Utilities (`escale`)**: Tools for efficient sharding and distributed computation in JAX.
- **Custom PyTrees (`jaximus`)**: Enhanced utilities for creating custom PyTrees and `ArrayValue` objects, updated from Equinox.
- **Custom Calling (`callib`)**: A tool for custom function calls and direct integration with Triton kernels in JAX.
- **Optimizer Factory**: A flexible factory for creating and configuring optimizers like AdamW, Adafactor, Lion, and RMSProp.
- **Custom Operations and Kernels**:
  - Flash Attention 2 for GPUs/TPUs (via Triton and Pallas).
  - 8-bit and NF4 quantization for efficient model.
- **Quantization Support**: Tools for 8-bit and NF4 quantization, enabling memory-efficient model deployment.

## Installation

You can install `eformer` via pip:

```bash
pip install eformer
```

## Quick Start

### Customizing Arrays With ArrayValue

```python
import jax

from eformer.jaximus import ArrayValue, implicit
from eformer.ops.quantization.quantization_functions import (
 dequantize_row_q8_0,
 quantize_row_q8_0,
)

array = jax.random.normal(jax.random.key(0), (256, 64), "f2")


class Array8B(ArrayValue):
 scale: jax.Array
 weight: jax.Array

 def __init__(self, array: jax.Array):
  self.weight, self.scale = quantize_row_q8_0(array)

 def materialize(self):
  return dequantize_row_q8_0(self.weight, self.scale)


qarray = Array8B(array)


@jax.jit
@implicit
def sqrt(x):
 return jax.numpy.sqrt(x)


print(sqrt(qarray))
print(qarray)

```

### Optimizer Factory

```python
from eformer.optimizers import OptimizerFactory, SchedulerConfig, AdamWConfig

# Create an AdamW optimizer with a cosine scheduler
scheduler_config = SchedulerConfig(scheduler_type="cosine", learning_rate=1e-3, steps=1000)
optimizer, scheduler = OptimizerFactory.create("adamw", scheduler_config, AdamWConfig())
```

### Quantization

```python
from eformer.quantization import Array8B, ArrayNF4

# Quantize an array to 8-bit
qarray = Array8B(jax.random.normal(jax.random.key(0), (256, 64), "f2"))

# Quantize an array to NF4
n4array = ArrayNF4(jax.random.normal(jax.random.key(0), (256, 64), "f2"), 64)
```

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
