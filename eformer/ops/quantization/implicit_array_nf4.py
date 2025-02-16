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

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union

import jax
from jax import lax
from jax import numpy as jnp

from eformer.jaximus import ImplicitArray, aux_field, register

from .quantization_functions import (
	dequantize_nf4,
	quantize_and_pack_nf4,
)

Array = jax.Array


@dataclass
class ArrayNF4(ImplicitArray):
	"""
	4-bit NormalFloat Quantization Class

	This class implements 4-bit NormalFloat (NF4) quantization for arrays. It quantizes the input array into 4-bit
	integers and stores the absolute maximum values for each block. The original array can be reconstructed using the
	stored packed data and absolute maximum values.

	Attributes:
	    packed (jax.Array): The packed 4-bit integer array.
	    absmax (jax.Array): The absolute maximum values for each block.
	    block_size (int): The size of each quantization block (static).

	Methods:
	    __init__(self, array: jax.Array, block_size: int = 64): Initializes the `ArrayNF4` object by quantizing the input array.
	    materialize(self): Reconstructs the original array from the quantized data.
	"""

	packed: Array
	absmax: Array
	block_size: int = aux_field()

	@classmethod
	def quantize(cls, array: Array, block_size: int = 64):
		"""
		Initializes the `ArrayNF4` object by quantizing the input array.

		Args:
		    array (jax.Array): The input array to be quantized.
		    block_size (int): The size of each quantization block. Defaults to 64.
		"""
		block_size = min(block_size, array.size)
		packed, absmax = quantize_and_pack_nf4(array, block_size)
		return cls(
			packed=packed,
			absmax=absmax,
			block_size=block_size,
			shape=array.shape,
			dtype=array.dtype,
		)

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
			.reshape(self.shape)
			.astype(self.dtype)
		)


ArrayType = Union[Array, ArrayNF4]


@register("convert_element_type")
def _(operand: ArrayType, new_dtype: Any) -> ArrayType:
	if isinstance(operand, ArrayNF4):
		operand.dtype = new_dtype
		return operand
	else:
		return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("lt")
def _(x: ArrayType, y: ArrayType, **kwargs):
	if isinstance(x, ArrayNF4):
		x = x.materialize()
	if isinstance(y, ArrayNF4):
		y = y.materialize()
	return jax.lax.lt(x, y, **kwargs)


@register("convert_element_type")
def _(operand: ArrayType, **kwargs) -> ArrayType:
	new_dtype = kwargs.get("new_dtype", jnp.bfloat16)
	if isinstance(operand, ArrayNF4):
		operand.dtype = new_dtype
		return operand
	else:
		return jax.lax.convert_element_type(operand=operand, new_dtype=new_dtype)


@register("integer_pow")
def _(x: Any, y: Any) -> Any:
	if isinstance(x, ArrayNF4):
		x = x.materialize()
	if isinstance(y, ArrayNF4):
		y = y.materialize()
	return lax.pow(x, y)


@register("integer_pow")
def _(x: Any, **kwargs) -> Any:
	y = kwargs.get("y", 2)
	if isinstance(x, ArrayNF4):
		x = x.materialize()
	return lax.pow(x, y)


@register("div")
def _(x: Any, y: Any) -> Any:
	if isinstance(x, ArrayNF4):
		x = x.materialize()
	if isinstance(y, ArrayNF4):
		y = y.materialize()
	return lax.div(x, y)


@register("sqrt")
def _(x: Any) -> Any:
	if isinstance(x, ArrayNF4):
		x = x.materialize()
	return lax.sqrt(x)


def safe_materialize(arr: ArrayType) -> Tuple[ArrayType, bool]:
	"""Safely materialize an array if it's ArrayNF4."""
	if isinstance(arr, ArrayNF4):
		arr = arr.materialize()
		return arr, True
	return arr, False


def safe_delete(arr: ArrayType, materialized: bool) -> None:
	"""Safely delete an array if it was materialized."""
	if materialized:
		del arr


@register("dot_general")
def handle_dot_general(
	lhs: ArrayType,
	rhs: ArrayType,
	*args: Any,
	**kwargs: Any,
) -> ArrayType:
	"""
	Custom handler for JAX's dot_general operation.

	Materializes ArrayNF4 inputs before performing the operation.

	Args:
	        primitive: The JAX primitive being handled.
	        lhs: Left-hand side array.
	        rhs: Right-hand side array.
	        *args: Variable length argument list.
	        **kwargs: Arbitrary keyword arguments.

	Returns:
	        The result of lax.dot_general operation.
	"""
	lhs, lhs_materialized = safe_materialize(lhs)
	rhs, rhs_materialized = safe_materialize(rhs)

	try:
		res = lax.dot_general(lhs, rhs, *args, **kwargs)
	finally:
		safe_delete(lhs, lhs_materialized)
		safe_delete(rhs, rhs_materialized)
	return res


@register("add")
def handle_add(x: ArrayType, y: ArrayType) -> ArrayType:
	"""
	Custom handler for JAX's add operation.

	Materializes ArrayNF4 inputs before performing the operation.

	Args:
	        primitive: The JAX primitive being handled.
	        x: First array to add.
	        y: Second array to add.

	Returns:
	        The result of lax.add operation.
	"""
	x, x_materialized = safe_materialize(x)
	y, y_materialized = safe_materialize(y)

	try:
		result = lax.add(x, y)
	finally:
		safe_delete(x, x_materialized)
		safe_delete(y, y_materialized)

	return result


@register("reduce")
def handle_reduce(
	operand: ArrayType,
	init_value: ArrayType,
	*args: Any,
	**kwargs: Any,
) -> ArrayType:
	"""
	Custom handler for JAX's reduce operation.

	Materializes ArrayNF4 inputs before performing the operation.

	Args:
	        primitive: The JAX primitive being handled.
	        operand: The array to be reduced.
	        init_value: The initial value for the reduction.
	        *args: Variable length argument list.
	        **kwargs: Arbitrary keyword arguments.

	Returns:
	        The result of lax.reduce operation.
	"""
	operand, operand_materialized = safe_materialize(operand)
	init_value, init_value_materialized = safe_materialize(init_value)

	try:
		result = lax.reduce(operand, init_value, *args, **kwargs)
	finally:
		safe_delete(operand, operand_materialized)
		safe_delete(init_value, init_value_materialized)

	return result


@register("mul")
def handle_mul(x: ArrayType, y: ArrayType) -> ArrayType:
	"""
	Custom handler for JAX's mul operation.

	Materializes ArrayNF4 inputs before performing the operation.

	Args:
	        primitive: The JAX primitive being handled.
	        x: First array to multiply.
	        y: Second array to multiply.

	Returns:
	        The result of lax.mul operation.
	"""
	x, x_materialized = safe_materialize(x)
	y, y_materialized = safe_materialize(y)

	try:
		result = lax.mul(x, y)
	finally:
		safe_delete(x, x_materialized)
		safe_delete(y, y_materialized)

	return result


@register("transpose")
def handle_transpose(
	operand: ArrayType,
	*args: Any,
	**kwargs: Any,
) -> ArrayType:
	"""
	Custom handler for JAX's transpose operation.

	Materializes ArrayNF4 input before performing the operation.
	Re-quantizes the result if the input was ArrayNF4.

	Args:
	        primitive: The JAX primitive being handled.
	        operand: The array to be transposed.
	        *args: Variable length argument list.
	        **kwargs: Arbitrary keyword arguments.

	Returns:
	        The result of lax.transpose operation, potentially re-quantized.
	"""
	operand, operand_materialized = safe_materialize(operand)

	try:
		result = lax.transpose(operand, *args, **kwargs)
	finally:
		safe_delete(operand, operand_materialized)

	return result


@register("conv_general_dilated")
def handle_conv(
	lhs: ArrayType,
	rhs: ArrayType,
	*args: Any,
	**kwargs: Any,
) -> ArrayType:
	"""
	Custom handler for JAX's conv_general_dilated operation.

	Materializes ArrayNF4 inputs before performing the operation.

	Args:
	        primitive: The JAX primitive being handled.
	        lhs: Left-hand side array (input).
	        rhs: Right-hand side array (kernel).
	        *args: Variable length argument list.
	        **kwargs: Arbitrary keyword arguments.

	Returns:
	        The result of lax.conv_general_dilated operation.
	"""
	lhs, lhs_materialized = safe_materialize(lhs)
	rhs, rhs_materialized = safe_materialize(rhs)

	try:
		result = lax.conv_general_dilated(lhs, rhs, *args, **kwargs)
	finally:
		safe_delete(lhs, lhs_materialized)
		safe_delete(rhs, rhs_materialized)

	return result


@register("max")
def handle_max(
	x: ArrayType,
	y: ArrayType,
	*args: Any,
	**kwargs: Any,
) -> ArrayType:
	"""
	Custom handler for JAX's max operation.

	Materializes ArrayNF4 inputs before performing the operation.

	Args:
	        primitive: The JAX primitive being handled.
	        x: First array for max comparison.
	        y: Second array for max comparison.
	        *args: Variable length argument list.
	        **kwargs: Arbitrary keyword arguments.

	Returns:
	        The result of lax.max operation.
	"""
	x, x_materialized = safe_materialize(x)
	y, y_materialized = safe_materialize(y)

	try:
		result = lax.max(x, y, *args, **kwargs)
	finally:
		safe_delete(x, x_materialized)
		safe_delete(y, y_materialized)

	return result


@register("exp")
def handle_exp(
	x: ArrayType,
	*args: Any,
	**kwargs: Any,
) -> ArrayType:
	"""
	Custom handler for JAX's exp operation.

	Materializes ArrayNF4 input before performing the operation.

	Args:
	        primitive: The JAX primitive being handled.
	        x: The array to apply exponential to.
	        *args: Variable length argument list.
	        **kwargs: Arbitrary keyword arguments.

	Returns:
	        The result of lax.exp operation.
	"""
	x, x_materialized = safe_materialize(x)

	try:
		result = lax.exp(x, *args, **kwargs)
	finally:
		safe_delete(x, x_materialized)

	return result


@register("log")
def handle_log(
	x: ArrayType,
	**kwargs: Any,
) -> jnp.ndarray:
	"""
	Custom handler for JAX's log operation.

	This function computes the natural logarithm of the input, handling both
	regular arrays and ArrayNF4 quantized arrays.

	Args:
	        primitive: The JAX primitive being handled.
	        x: The array to apply logarithm to.
	        **kwargs: Additional keyword arguments for the log operation.

	Returns:
	        The result of the natural logarithm operation.

	Raises:
	        RuntimeError: If the log operation fails.
	"""
	x, x_materialized = safe_materialize(x)

	try:
		result = lax.log(x, **kwargs)
	except Exception as e:
		raise RuntimeError(f"Log operation failed: {str(e)}") from e
	finally:
		safe_delete(x, x_materialized)

	return result


@register("reshape")
def handle_reshape(
	operand: ArrayType,
	**kwargs: Any,
) -> ArrayType:
	"""
	Custom handler for JAX's reshape operation.

	This function handles reshaping for both regular arrays and ArrayNF4 quantized arrays.
	It materializes ArrayNF4 input before reshaping and re-quantizes the result if the input was ArrayNF4.

	Args:
	        primitive: The JAX primitive being handled.
	        operand: The array to be reshaped.
	        **kwargs: Additional keyword arguments for the reshape operation.

	Returns:
	        The reshaped array, potentially re-quantized if the input was ArrayNF4.

	Raises:
	        ValueError: If the new shape is not compatible with the original array's size.
	"""
	operand, operand_materialized = safe_materialize(operand)

	try:
		reshaped = lax.reshape(operand, **kwargs)
	except ValueError as e:
		raise ValueError(
			f"Reshape operation failed: {str(e)}. "
			f"Ensure the new shape {kwargs} is compatible with the original array size."
		) from e
	finally:
		safe_delete(operand, operand_materialized)

	return reshaped


@register("concatenate")
def handle_concatenate(
	operands: Sequence[ArrayType],
	*args: Any,
	**kwargs: Any,
) -> ArrayType:
	"""
	Custom handler for JAX's concatenate operation.

	Materializes ArrayNF4 inputs before performing the operation.

	Args:
	                primitive: The JAX primitive being handled.
	                operands: Sequence of arrays to concatenate.
	                *args: Variable length argument list.
	                **kwargs: Arbitrary keyword arguments.

	Returns:
	                The result of lax.concatenate operation.
	"""
	materialized_operands = []
	materialized_flags = []

	for op in operands:
		mat_op, mat_flag = safe_materialize(op)
		materialized_operands.append(mat_op)
		materialized_flags.append(mat_flag)

	try:
		result = lax.concatenate(materialized_operands, *args, **kwargs)
	finally:
		for op, flag in zip(materialized_operands, materialized_flags):  # noqa
			safe_delete(op, flag)

	return result


@register("broadcast_in_dim")
def handle_broadcast_in_dim(
	operand: ArrayType,
	*args: Any,
	**kwargs: Any,
) -> ArrayType:
	"""Handle broadcast_in_dim for ArrayNF4."""
	operand, operand_materialized = safe_materialize(operand)

	try:
		result = jax.lax.broadcast_in_dim(operand, *args, **kwargs)
	finally:
		safe_delete(operand, operand_materialized)

	return result


@register("gather")
def handle_gather(
	operand: ArrayType,
	*args: Any,
	**kwargs: Any,
) -> ArrayType:
	"""Handle gather for ArrayNF4."""
	operand, operand_materialized = safe_materialize(operand)

	try:
		result = jax.lax.gather(operand, *args, **kwargs)
	finally:
		safe_delete(operand, operand_materialized)

	return result
