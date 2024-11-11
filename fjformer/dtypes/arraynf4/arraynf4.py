# NF4 impl from QLoRA Paper

from dataclasses import dataclass
from functools import partial
import functools
from typing import Any, Optional, Sequence, Tuple, Union

import chex
import jax
from jax import Array, lax
from jax import numpy as jnp
from jax.core import Primitive

import fjformer.core as core

BLOCK_SIZE = 1024

nf4_table = jnp.array(
	[
		-1.0,  # 0000
		-0.6961928009986877,  # 0001
		-0.5250730514526367,  # 0010
		-0.39491748809814453,  # 0011
		-0.28444138169288635,  # 0100
		-0.18477343022823334,  # 0101
		-0.09105003625154495,  # 0110
		0.0,  # 0111
		0.07958029955625534,  # 1000
		0.16093020141124725,  # 1001
		0.24611230194568634,  # 1010
		0.33791524171829224,  # 1011
		0.44070982933044434,  # 1100
		0.5626170039176941,  # 1101
		0.7229568362236023,  # 1110
		1.0,  # 1111
	],
	dtype=jnp.float32,
)


@jax.jit
def dequantize_nf4_value(val):
	"""
	Dequantizes a single NF4 value (4-bit, 0-15) to its corresponding float value.
	"""
	# Convert the lookup table into a more JAX-friendly format
	return nf4_table[val]


@jax.jit
def quantize_nf4(x):
	"""
	Implements NF4 quantization using JAX with optimized array operations.
	Input should be normalized (divided by absmax) before calling this function.
	Returns values in the range 0-15 (4 bits).
	"""
	# Define the boundary points for quantization
	boundaries = jnp.array(
		[
			-float("inf"),  # 0
			-0.8480964004993439,  # 1
			-0.6106329262256622,  # 2
			-0.4599952697753906,  # 3
			-0.33967943489551544,  # 4
			-0.23460740596055984,  # 5
			-0.13791173323988914,  # 6
			-0.045525018125772476,  # 7
			0.03979014977812767,  # 8
			0.1202552504837513,  # 9
			0.2035212516784668,  # 10
			0.2920137718319893,  # 11
			0.3893125355243683,  # 12
			0.5016634166240692,  # 13
			0.6427869200706482,  # 14
			0.8614784181118011,  # 15
		],
		dtype=jnp.float32,
	)

	# Use searchsorted to find the appropriate quantization level
	indices = jnp.searchsorted(boundaries, x)
	# Subtract 1 because searchsorted returns the insertion point
	return indices - 1


@jax.jit
def pack_nf4_pairs(x):
	"""
	Packs pairs of NF4 values (0-15) into single bytes.
	First value goes into the high bits, second into the low bits.
	"""
	x = x.reshape(-1, 2)
	return (x[:, 0] << 4) | x[:, 1]


@functools.partial(jax.jit, static_argnames=["block_size"])
def quantize_array_nf4(x, block_size=BLOCK_SIZE):
	"""
	Quantizes an array using NF4 quantization.

	Args:
	    x: Input array
	    block_size: Size of blocks for computing absolute maximum

	Returns:
	    packed_values: Packed quantized values (uint8)
	    absmax: Absolute maximum values for each block
	"""
	# Reshape input into blocks
	pad_size = (block_size - (x.size % block_size)) % block_size
	padded_x = jnp.pad(x, (0, pad_size))
	blocks = padded_x.reshape(-1, block_size)

	# Compute absolute maximum for each block
	absmax = jnp.max(jnp.abs(blocks), axis=1)

	# Normalize and quantize
	normalized = blocks / absmax[:, None]
	quantized = quantize_nf4(normalized.reshape(-1)).reshape(-1)

	# Pack pairs of values into bytes
	packed = pack_nf4_pairs(quantized)

	return packed, absmax


@jax.jit
def unpack_nf4_pairs(packed_vals):
	"""
	Unpacks bytes containing two NF4 values into separate 4-bit values.
	Returns two arrays: high bits and low bits.
	"""
	high = (packed_vals >> 4) & 0xF
	low = packed_vals & 0xF
	return jnp.stack([high, low], axis=1).reshape(-1)


@functools.partial(jax.jit, static_argnames=["block_size"])
def dequantize_array_nf4(packed_values, absmax, block_size=BLOCK_SIZE):
	"""
	Dequantizes packed NF4 values back to full precision.

	Args:
	    packed_values: Packed quantized values (uint8)
	    absmax: Absolute maximum values for each block
	    original_size: Original size of the array before quantization

	Returns:
	    Dequantized array of original size
	"""
	# Unpack the 4-bit values
	unpacked = unpack_nf4_pairs(packed_values)

	# Dequantize the values
	dequantized = jax.vmap(dequantize_nf4_value)(unpacked)

	# Calculate number of complete blocks
	num_blocks = len(absmax)

	# Reshape dequantized values to match blocks
	dequantized = dequantized.reshape(num_blocks, block_size)

	# Scale each block by its corresponding absmax
	scaled = dequantized * absmax[:, None]

	# Reshape back to original size
	result = jax.lax.dynamic_slice_in_dim(
		scaled.reshape(-1),
		0,
		unpacked.shape[0],
		0,
	)

	return result


@dataclass
class ArrayNF4(core.ImplicitArray):
	packed: core.ArrayValue
	absmax: core.ArrayValue
	block_size: int = core.aux_field()

	def materialize(self) -> Array:
		"""
		Materialize the 4-bit array into a full-precision array.

		Returns:
		        Array: The dequantized array.
		"""
		return self.dequantize()

	def dequantize(self, dtype: Optional[jnp.dtype] = None) -> Array:
		"""
		Dequantize the 4-bit array into a full-precision array.

		Args:
		        dtype (Optional[jnp.dtype]): Desired dtype of the output array.

		Returns:
		        Array: Dequantized array.
		"""

		dtype = dtype if dtype is not None else self.dtype
		return (
			dequantize_array_nf4(self.packed, self.absmax, self.block_size)
			.reshape(self.shape)
			.astype(dtype)
		)

	@classmethod
	def quantize(cls, array: chex.Array) -> "ArrayNF4":
		shape = array.shape
		bs = array.size // 4
		(packed, absmax) = quantize_array_nf4(array.reshape(-1), bs)
		return cls(
			packed=packed,
			absmax=absmax,
			dtype=array.dtype,
			shape=shape,
			block_size=bs,
		)


ArrayType = Union[Array, ArrayNF4]


def safe_materialize(arr: ArrayType) -> Tuple[ArrayType, bool]:
	"""Safely materialize an array if it's ArrayNF4."""
	if isinstance(arr, ArrayNF4):
		return arr.materialize(), True
	return arr, False


def safe_delete(arr: ArrayType, materialized: bool) -> None:
	"""Safely delete an array if it was materialized."""
	if materialized:
		del arr


@core.primitive_handler("dot_general")
def handle_dot_general(
	primitive: Primitive,
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


@core.primitive_handler("add")
def handle_add(primitive: Primitive, x: ArrayType, y: ArrayType) -> ArrayType:
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


@core.primitive_handler("reduce")
def handle_reduce(
	primitive: Primitive,
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


@core.primitive_handler("mul")
def handle_mul(primitive: Primitive, x: ArrayType, y: ArrayType) -> ArrayType:
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


@core.primitive_handler("transpose")
def handle_transpose(
	primitive: Primitive,
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


@core.primitive_handler("conv_general_dilated")
def handle_conv(
	primitive: Primitive,
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


@core.primitive_handler("max")
def handle_max(
	primitive: Primitive,
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


@core.primitive_handler("exp")
def handle_exp(
	primitive: Primitive,
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


@core.primitive_handler("log")
def handle_log(
	primitive: Primitive,
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


@core.primitive_handler("reshape")
def handle_reshape(
	primitive: Primitive,
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


@core.primitive_handler("concatenate")
def handle_concatenate(
	primitive: Primitive,
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


@core.primitive_handler("convert_element_type")
def convert_element_type(
	primitive: Primitive,
	arg: ArrayNF4,
	**params: Any,
) -> ArrayNF4:
	"""Handle element type conversion for ArrayNF4."""
	result = jax.tree_util.tree_map(
		partial(
			core.default_handler,
			primitive,
			**params,
		),
		arg,
	)
	result.dtype = params["new_dtype"]
	return result


def _out_shape_dtype(
	primitive: Any,
	*args: Any,
	**kwargs: Any,
) -> Tuple[Tuple[int, ...], Any]:
	"""
	Determine the output shape and dtype for a given primitive operation.

	This function uses JAX's shape inference capabilities to determine what
	the shape and dtype of the output would be if the operation were performed
	on concrete arrays.

	Args:
	                primitive: JAX primitive to be evaluated.
	                *args: Positional arguments for the primitive.
	                **kwargs: Keyword arguments for the primitive.

	Returns:
	                A tuple containing:
	                                - The shape of the output as a tuple of integers.
	                                - The dtype of the output.
	"""
	out_aval = jax.eval_shape(
		partial(core.default_handler, primitive, **kwargs),
		*(jax.core.get_aval(x) for x in args),
	)
	return jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), out_aval)


@core.primitive_handler(
	[
		"reshape",
		"broadcast_in_dim",
		"reduce_min",
		"reduce_max",
		"reduce_or",
		"reduce_and",
	]
)
def unchanged_value_op(
	primitive: Any,
	sym: core.symbols.SymbolicConstant,
	**kwargs: Any,
) -> core.symbols.SymbolicConstant:
	"""
	Handler for JAX primitives that don't change the value of a SymbolicConstant.

	This function handles operations that may change the shape or other properties
	of a SymbolicConstant, but not its fundamental value.

	Args:
	                primitive: The JAX primitive being handled.
	                sym: The symbolic constant being operated on.
	                **kwargs: Additional keyword arguments for the primitive operation.

	Returns:
	                A new SymbolicConstant with potentially updated shape and dtype, but the same value as the input.
	"""
	out_shape, out_dtype = _out_shape_dtype(primitive, sym, **kwargs)
	return core.symbols.SymbolicConstant(sym.value, shape=out_shape, dtype=out_dtype)


@core.primitive_handler("broadcast_in_dim")
def handle_broadcast_in_dim(
	primitive: Primitive,
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


@core.primitive_handler("gather")
def handle_gather(
	primitive: Primitive,
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
