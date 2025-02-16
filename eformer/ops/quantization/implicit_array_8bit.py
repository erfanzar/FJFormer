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

These classes are designed to reduce memory usage and computational overhead while maintaining reasonable accuracy for
machine learning models. They are built on top of JAX, a high-performance numerical computing library.

Classes:
    - `Array8B`: Implements 8-bit quantization for arrays.

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

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import jax
from jax import lax
from jax import numpy as jnp
from jax.extend.core import Primitive

from eformer.jaximus import ImplicitArray, register

from .quantization_functions import dequantize_row_q8_0, quantize_row_q8_0

Array = jax.Array


@dataclass
class Array8B(ImplicitArray):
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

	@classmethod
	def quantize(cls, array: Array, dtype: Optional[jnp.dtype] = None):
		"""
		Initializes the `Array8B` object by quantizing the input array.

		Args:
		    array (jax.Array): The input array to be quantized.
		"""
		weight, scale = quantize_row_q8_0(array)
		return cls(
			weight=weight,
			scale=scale,
			shape=array.shape,
			dtype=dtype or array.dtype,
		)

	def materialize(self):
		"""
		Reconstructs the original array from the quantized data.

		Returns:
		    jax.Array: The dequantized array.
		"""
		return (
			dequantize_row_q8_0(
				self.weight,
				self.scale,
			)
			.reshape(self.shape)
			.astype(self.dtype)
		)


ArrayType = Union[Array, Array8B]


@register("lt")
def _(x: ArrayType, y: ArrayType, **kwargs): 
	if isinstance(x, Array8B):
		x = x.materialize()
	if isinstance(y, Array8B):
		y = y.materialize()
	return jax.lax.lt(x, y, **kwargs) 


@register("convert_element_type")
def _(operand: ArrayType, new_dtype: Any) -> ArrayType:
	if isinstance(operand, Array8B):
		operand.dtype = new_dtype
		return operand
	else:
		return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("convert_element_type")
def _(operand: ArrayType, **kwargs) -> ArrayType:
	new_dtype = kwargs.get("new_dtype", jnp.bfloat16)
	if isinstance(operand, Array8B):
		operand.dtype = new_dtype
		return operand
	else:
		return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("integer_pow")
def _(x: Any, y: Any) -> Any:
	if isinstance(x, Array8B):
		x = x.materialize()
	if isinstance(y, Array8B):
		y = y.materialize()
	return lax.pow(x, y)


@register("integer_pow")
def _(x: Any, **kwargs) -> Any:
	y = kwargs.get("y", 2)
	if isinstance(x, Array8B):
		x = x.materialize()
	return lax.pow(x, y)


@register("div")
def _(x: Any, y: Any) -> Any:
	if isinstance(x, Array8B):
		x = x.materialize()
	if isinstance(y, Array8B):
		y = y.materialize()
	return lax.div(x, y)


@register("sqrt")
def _(x: Any) -> Any:
	if isinstance(x, Array8B):
		x = x.materialize()
	return lax.sqrt(x)


@register("dot_general")
def handle_dot_general(
	lhs: ArrayType,
	rhs: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's dot_general operation.

	Materializes Array8B inputs before performing the operation.

	Args:

	    lhs (ArrayType): Left-hand side array.
	    rhs (ArrayType): Right-hand side array.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.dot_general operation.
	"""
	if isinstance(lhs, Array8B):
		lhs = lhs.materialize()
	if isinstance(rhs, Array8B):
		rhs = rhs.materialize()
	return lax.dot_general(lhs, rhs, *args, **kwargs)


@register("add")
def handle_add(
	x: ArrayType,
	y: ArrayType,
):
	"""
	Custom handler for JAX's add operation.

	Materializes Array8B inputs before performing the operation.

	Args:

	    x (ArrayType): First array to add.
	    y (ArrayType): Second array to add.

	Returns:
	    The result of lax.add operation.
	"""
	if isinstance(x, Array8B):
		x = x.materialize()
	if isinstance(y, Array8B):
		y = y.materialize()
	return lax.add(x, y)


@register("reduce")
def handle_reduce(
	operand: ArrayType,
	init_value: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's reduce operation.

	Materializes Array8B inputs before performing the operation.

	Args:

	    operand (ArrayType): The array to be reduced.
	    init_value (ArrayType): The initial value for the reduction.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.reduce operation.
	"""

	if isinstance(operand, Array8B):
		operand = operand.materialize()
	if isinstance(init_value, Array8B):
		init_value = init_value.materialize()
	return lax.reduce(operand, init_value, *args, **kwargs)


@register("mul")
def handle_mul(
	x: ArrayType,
	y: ArrayType,
):
	"""
	Custom handler for JAX's mul operation.

	Materializes Array8B inputs before performing the operation.

	Args:

	    x (ArrayType): First array to multiply.
	    y (ArrayType): Second array to multiply.

	Returns:
	    The result of lax.mul operation.
	"""
	if isinstance(x, Array8B):
		x = x.materialize()
	if isinstance(y, Array8B):
		y = y.materialize()
	return lax.mul(x, y)


@register("transpose")
def handle_transpose(
	operand: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's transpose operation.

	Materializes Array8B input before performing the operation.
	Re-quantizes the result if the input was Array8B.

	Args:

	    operand (ArrayType): The array to be transposed.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.transpose operation, potentially re-quantized.
	"""
	original_quantized = False
	if isinstance(operand, Array8B):
		operand = operand.materialize()
		original_quantized = True
	operand = lax.transpose(operand, *args, **kwargs)
	if original_quantized:
		operand = Array8B.quantize(operand, dtype=operand.dtype)
	return operand


@register("conv_general_dilated")
def handle_conv(
	lhs: ArrayType,
	rhs: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's conv_general_dilated operation.

	Materializes Array8B inputs before performing the operation.

	Args:

	    lhs (ArrayType): Left-hand side array (input).
	    rhs (ArrayType): Right-hand side array (kernel).
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.conv operation.
	"""
	if isinstance(lhs, Array8B):
		lhs = lhs.materialize()
	if isinstance(rhs, Array8B):
		rhs = rhs.materialize()
	return lax.conv_general_dilated(lhs, rhs, *args, **kwargs)


@register("max")
def handle_max(
	x: ArrayType,
	y: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's max operation.

	Materializes Array8B inputs before performing the operation.

	Args:

	    x (ArrayType): First array for max comparison.
	    y (ArrayType): Second array for max comparison.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.max operation.
	"""
	if isinstance(x, Array8B):
		x = x.materialize()
	if isinstance(y, Array8B):
		y = y.materialize()
	return lax.max(x, y, *args, **kwargs)


@register("exp")
def handle_exp(
	x: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's exp operation.

	Materializes Array8B input before performing the operation.

	Args:

	    x (ArrayType): The array to apply exponential to.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.exp operation.
	"""
	if isinstance(x, Array8B):
		x = x.materialize()
	return lax.exp(x, *args, **kwargs)


@register("log")
def handle_log(
	x: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's log operation.

	Materializes Array8B input before performing the operation.

	Args:

	    x (ArrayType): The array to apply logarithm to.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.log operation.
	"""
	if isinstance(x, Array8B):
		x = x.materialize()
	return lax.log(x, *args, **kwargs)


@register("reshape")
def handle_reshape(
	primitive: Primitive,
	operand: ArrayType,
	**kwargs: Any,
):
	"""
	Custom handler for JAX's reshape operation.

	This function handles reshaping for both regular arrays and Array8B quantized arrays.
	It materializes ArrayNF4 input before reshaping and re-quantizes the result if the input was ArrayNF4.

	Args:
	    primitive (Primitive): The JAX primitive being handled.
	    operand (ArrayType): The array to be reshaped.
	    new_sizes (Tuple[int, ...]): The desired new shape of the array.
	    dimensions (Tuple[int, ...], optional): The order in which dimensions should be permuted before reshaping.
	    **kwargs: Additional keyword arguments for the reshape operation.

	Returns:
	    ArrayType: The reshaped array, potentially re-quantized if the input was Array8B.

	Raises:
	    ValueError: If the new shape is not compatible with the original array's size.
	"""
	original_quantized = isinstance(operand, Array8B)

	if original_quantized:
		operand = operand.materialize()

	try:
		reshaped = lax.reshape(operand, **kwargs)
	except ValueError as e:
		raise ValueError(
			f"Reshape operation failed: {str(e)}. "
			f"Ensure the new shape {kwargs} is compatible with the original array size."
		) from e
	if original_quantized:
		reshaped = Array8B.quantize(reshaped, dtype=reshaped.dtype)
	return reshaped


@register("concatenate")
def handle_concatenate(
	operands: Sequence[ArrayType],
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's concatenate operation.

	Materializes Array8B inputs before performing the operation.

	Args:

	    operands (Sequence[ArrayType]): Sequence of arrays to concatenate.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.concatenate operation.
	"""
	materialized_operands = [
		op.materialize() if isinstance(op, Array8B) else op for op in operands
	]
	return lax.concatenate(materialized_operands, *args, **kwargs)


@register("broadcast_in_dim")
def handle_broadcast_in_dim(
	operand: ArrayType,
	*args,
	**kwargs,
) -> ArrayType:
	"""Handle broadcast_in_dim for Array8B."""
	original_quantized = isinstance(operand, Array8B)
	array = operand
	if original_quantized:
		array = operand.materialize()
	result = jax.lax.broadcast_in_dim(array, *args, **kwargs)
	if original_quantized:
		result = Array8B.quantize(result, dtype=operand.dtype)
	return result


@register("gather")
def handle_gather(
	operand: ArrayType,
	*args,
	**kwargs,
) -> ArrayType:
	"""Handle gather for Array8B."""
	original_quantized = isinstance(operand, Array8B)
	array = operand
	if original_quantized:
		array = operand.materialize()
	result = jax.lax.gather(array, *args, **kwargs)
	return result
