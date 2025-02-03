# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Quantization Module

This module provides functionality for quantizing and dequantizing arrays using two different quantization methods:
- 8-bit quantization (`Array8B`)
- 4-bit NormalFloat quantization (`ArrayNF4`)

These classes are designed to reduce memory usage and computational overhead while maintaining reasonable accuracy for
machine learning models. They are built on top of JAX, a high-performance numerical computing library.

Classes:
    - `Array8B`: Implements 8-bit quantization for arrays.
    - `ArrayNF4`: Implements 4-bit NormalFloat quantization for arrays.

Usage Example:
    ```python
    import jax
    from eformer.ops.quantization import Array8B, ArrayNF4
    from eformer.jaximus import implicit

    array = jax.random.normal(jax.random.key(0), (256, 64), "f2")

    # Quantize the array using 8-bit quantization
    qarray = Array8B(array)

    # Quantize the array using 4-bit NormalFloat quantization
    n4array = ArrayNF4(array)


    # Define a function to apply to the quantized arrays
    def power(x):
      return x**2


    # Apply the function to the quantized arrays
    print(jax.jit(implicit(power))(qarray))
    print(qarray)

    print(jax.jit(implicit(power))(n4array))
    print(n4array)
    ```
"""

from eformer.jaximus import ArrayValue
from eformer.jaximus._core import field
from .quantization_functions import (
	dequantize_nf4,
	quantize_and_pack_nf4,
	quantize_row_q8_0,
	dequantize_row_q8_0,
)
import jax
from jax import numpy as jnp

Array = jax.Array


class Array8B(ArrayValue):
	"""
	8-bit Quantization Class

	This class implements 8-bit quantization for arrays. It quantizes the input array into 8-bit integers and stores
	the quantization scale factor. The original array can be reconstructed (dequantized) using the stored scale factor.

	Attributes:
	    scale (jax.Array): The scale factor used for quantization.
	    weight (jax.Array): The quantized 8-bit integer array.

	Methods:
	    __init__(self, array: jax.Array): Initializes the `Array8B` object by quantizing the input array.
	    materialize(self): Reconstructs the original array from the quantized data.
	"""

	scale: Array
	weight: Array

	def __init__(self, array: Array):
		"""
		Initializes the `Array8B` object by quantizing the input array.

		Args:
		    array (jax.Array): The input array to be quantized.
		"""
		self.weight, self.scale = quantize_row_q8_0(array)

	def materialize(self):
		"""
		Reconstructs the original array from the quantized data.

		Returns:
		    jax.Array: The dequantized array.
		"""
		return dequantize_row_q8_0(self.weight, self.scale)


class ArrayNF4(ArrayValue):
	"""
	4-bit NormalFloat Quantization Class

	This class implements 4-bit NormalFloat (NF4) quantization for arrays. It quantizes the input array into 4-bit
	integers and stores the absolute maximum values for each block. The original array can be reconstructed using the
	stored packed data and absolute maximum values.

	Attributes:
	    packed (jax.Array): The packed 4-bit integer array.
	    absmax (jax.Array): The absolute maximum values for each block.
	    _ashape (tuple): The shape of the original array (static).
	    _adtype (jnp.dtype): The data type of the original array (static).
	    block_size (int): The size of each quantization block (static).

	Methods:
	    __init__(self, array: jax.Array, block_size: int = 64): Initializes the `ArrayNF4` object by quantizing the input array.
	    materialize(self): Reconstructs the original array from the quantized data.
	"""

	packed: Array
	absmax: Array
	_ashape: tuple = field(static=True)
	_adtype: jnp.dtype = field(static=True)
	block_size: int = field(static=True)

	def __init__(self, array: Array, block_size: int = 64):
		"""
		Initializes the `ArrayNF4` object by quantizing the input array.

		Args:
		    array (jax.Array): The input array to be quantized.
		    block_size (int): The size of each quantization block. Defaults to 64.
		"""
		self.block_size = min(block_size, array.size)
		self._ashape = array.shape
		self._adtype = array.dtype
		self.packed, self.absmax = quantize_and_pack_nf4(array, block_size)

	def materialize(self):
		"""
		Reconstructs the original array from the quantized data.

		Returns:
		    jax.Array: The dequantized array.
		"""
		return (
			dequantize_nf4(
				self.packed.astype(jnp.uint8),
				self.absmax,
				self.block_size,
			)
			.reshape(self._ashape)
			.astype(self._adtype)
		)
