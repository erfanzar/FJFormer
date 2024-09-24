# TODO : Implement Custom Backward Prp
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import jax
from jax import Array, lax
from jax import numpy as jnp
from jax.core import Primitive

import fjformer.core as core


def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
	"""
	Creates the dynamic quantization map.
	"""

	data = []
	non_sign_bits = total_bits - (1 if signed else 0)
	additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
	for i in range(max_exponent_bits):
		fraction_items = int(
			(
				2 ** (i + non_sign_bits - max_exponent_bits) + 1
				if signed
				else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1
			),
		)
		boundaries = jnp.linspace(0.1, 1, fraction_items)
		means = (boundaries[:-1] + boundaries[1:]) / 2.0
		data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
		if signed:
			data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

	if additional_items > 0:
		boundaries = jnp.linspace(0.1, 1, additional_items + 1)
		means = (boundaries[:-1] + boundaries[1:]) / 2.0
		data += (
			(10 ** (-(max_exponent_bits - 1) + (max_exponent_bits - 1))) * means
		).tolist()
		if signed:
			data += (
				-(10 ** (-(max_exponent_bits - 1) + (max_exponent_bits - 1))) * means
			).tolist()

	data.append(0)
	data.append(1.0)

	assert len(data) == 2**total_bits

	gap = 256 - len(data)
	for _ in range(gap):
		data.append(0)

	data.sort()
	return jnp.array(data, dtype=jnp.float32)


CODE = create_dynamic_map()
CODE = CODE.at[0].set(-1.0)
OLD_QUANT = True
DEFAULT_BLOCKSIZE = 1024


@jax.jit
def quantize_block_jax(block):
	# 1. find absmax in block
	absmax_block = jnp.max(jnp.abs(block))

	# 2. divide input value by absmax to normalize into [-1.0, 1.0]
	normed_values = block / absmax_block

	# 3. do binary search to find the closest value
	idx = jnp.searchsorted(CODE, normed_values)

	# 4. check minimal distance
	def check_distance(idx, normed_value):
		dist_left = jnp.abs(normed_value - CODE[idx])
		dist_right = jnp.abs(normed_value - CODE[idx + 1])
		return jnp.where(
			idx < 255,
			jnp.where(dist_right < dist_left, idx + 1, idx),
			idx,
		)

	idx = jax.vmap(check_distance)(idx, normed_values)

	# 5. store index
	return idx.astype(jnp.uint8), absmax_block


@functools.partial(jax.jit, static_argnames=["blocksize"])
def quantize_jax(A, blocksize):
	# Ensure code[0] is -1.0

	# Flatten the input array
	A_flat = A.ravel()

	# Pad A_flat to be divisible by blocksize
	pad_size = (blocksize - A_flat.size % blocksize) % blocksize
	A_padded = jnp.pad(A_flat, (0, pad_size))

	# Reshape A_padded into blocks
	num_blocks = A_padded.size // blocksize
	A_blocks = A_padded.reshape(num_blocks, blocksize)

	# Apply quantize_block to each block
	quantized_blocks, absmax = jax.vmap(quantize_block_jax, in_axes=(0,))(A_blocks)

	# Flatten the quantized blocks and trim to original size
	quantized = quantized_blocks.ravel()[: A_flat.size]

	# Reshape quantized back to the original shape
	quantized = quantized.reshape(A.shape)

	return quantized, absmax.astype(jnp.float32)


if OLD_QUANT:

	@functools.partial(jax.jit, static_argnames=["blocksize"])
	def dequantize_jax(A, absmax, blocksize):
		"""
		Dequantize the input array.

		Args:
		    A (jnp.ndarray): The quantized array.
		    absmax (jnp.ndarray): The absolute maximum values for each block.
		    blocksize (int): The size of the blocks for dequantization.

		Returns:
		    jnp.ndarray: The dequantized array.
		"""
		indices = jnp.arange(A.shape[0])
		block_indices = indices // blocksize
		extra_dim = ()
		if CODE[A].ndim > 1:
			extra_dim = (1,) * (len(CODE[A].shape) - 1)
		dequantized_A = CODE[A] * jnp.reshape(
			absmax[block_indices],
			(CODE[A].shape[0],) + extra_dim,
		)
		return dequantized_A

else:

	@functools.partial(jax.jit, static_argnames=["blocksize"])
	def dequantize_jax(A, absmax, blocksize):
		# Flatten A
		A_flat = A.ravel()

		# Compute the number of blocks
		num_blocks = (A_flat.size + blocksize - 1) // blocksize

		# Pad A to be divisible by blocksize
		pad_size = (blocksize - A_flat.size % blocksize) % blocksize
		A_padded = jnp.pad(A_flat, (0, pad_size))

		# Reshape A_padded into blocks
		A_blocks = A_padded.reshape(num_blocks, blocksize)

		# Ensure absmax has the correct shape
		absmax = absmax[:num_blocks]

		# Define the dequantization function for a single block
		def dequantize_block(block, block_absmax):
			return CODE[block] * block_absmax

		# Apply dequantize_block to each block
		out_blocks = jax.vmap(dequantize_block)(A_blocks, absmax)

		# Flatten and trim to original size
		out = out_blocks.ravel()[: A_flat.size]

		# Reshape out to match the original shape of A
		out = out.reshape(A.shape)

		return out


@dataclass
class Array8Lt(core.ImplicitArray):
	"""
	Custom 8-bit Quantized Array for efficient manipulation of JAX arrays.

	This class provides methods for quantizing and dequantizing JAX arrays to 8-bit
	representation, which can significantly reduce memory usage and potentially
	improve computation speed for certain operations.

	Example:
	    >>> import jax
	    >>> import jax.numpy as jnp

	    >>> x = jax.random.normal(jax.random.key(0), (512, 64), dtype=jnp.float32)
	    >>> xp = jax.random.normal(jax.random.key(1), (64, 256), dtype=jnp.float32)

	    >>> quantized_x = Array8Lt.quantize(
	    ...   x,
	    ... )
	    >>> quantized_xp = Array8Lt.quantize(
	    ...   xp,
	    ... )

	    >>> @jax.jit
	    >>> @core.implicit_compact
	    >>> def f(a, b):
	    ...   return jnp.dot(a, b)

	    >>> result = f(x, xp)
	    >>> q_result = f(quantized_x, quantized_xp)

	    >>> print(jnp.allclose(result, q_result, rtol=1e-2, atol=1e-2))
	    True
	"""

	A: core.ArrayValue
	absmax: core.ArrayValue
	blocksize: int = core.aux_field()

	def materialize(self) -> Array:
		"""
		Materialize the quantized array back to its original representation.

		Returns:
		    Array: The dequantized array in its original dtype.
		"""
		return self.dequantize(
			A=self.A,
			absmax=self.absmax,
			blocksize=self.blocksize,
			float_dtype=self.dtype,
		)

	@classmethod
	def quantize(
		cls,
		array: Array,
		blocksize: int = DEFAULT_BLOCKSIZE,
		dtype: Optional[jnp.dtype] = None,
	) -> Array8Lt:
		QA, absmax = quantize_jax(array, blocksize)
		return cls(
			A=QA,
			absmax=absmax,
			shape=QA.shape,
			blocksize=blocksize,
			dtype=dtype or array.dtype,
		)

	@staticmethod
	def dequantize(
		A,
		absmax,
		blocksize,
		float_dtype: jnp.dtype,
	) -> Array:
		return dequantize_jax(
			A=A,
			absmax=absmax.astype(jnp.float32),
			blocksize=blocksize,
		).astype(float_dtype)

	def __repr__(self) -> str:
		return f"Array8Lt(shape={self.shape}, dtype={self.dtype})"

	@property
	def nbytes(self) -> int:
		"""
		Calculate the total number of bytes used by the quantized representation.

		Returns:
		    int: The number of bytes used.
		"""
		return self.A.nbytes + self.absmax.nbytes

	def memory_savings(self) -> float:
		"""
		Calculate the memory savings compared to the original array.

		Returns:
		    float: The percentage of memory saved.
		"""
		original_size = jnp.prod(jnp.array(self.shape)) * jnp.dtype(self.dtype).itemsize
		return (1 - self.nbytes / original_size) * 100


ArrayType = Union[Array, Array8Lt]


@core.primitive_handler("dot_general")
def handle_dot_general(
	primitive,
	lhs: ArrayType,
	rhs: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's dot_general operation.

	Materializes Array8Lt inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    lhs (ArrayType): Left-hand side array.
	    rhs (ArrayType): Right-hand side array.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.dot_general operation.
	"""
	if isinstance(lhs, Array8Lt):
		lhs = lhs.materialize()
	if isinstance(rhs, Array8Lt):
		rhs = rhs.materialize()
	return lax.dot_general(lhs, rhs, *args, **kwargs)


@core.primitive_handler("add")
def handle_add(
	primitive,
	x: ArrayType,
	y: ArrayType,
):
	"""
	Custom handler for JAX's add operation.

	Materializes Array8Lt inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): First array to add.
	    y (ArrayType): Second array to add.

	Returns:
	    The result of lax.add operation.
	"""
	if isinstance(x, Array8Lt):
		x = x.materialize()
	if isinstance(y, Array8Lt):
		y = y.materialize()
	return lax.add(x, y)


@core.primitive_handler("reduce")
def handle_reduce(
	primitive,
	operand: ArrayType,
	init_value: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's reduce operation.

	Materializes Array8Lt inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    operand (ArrayType): The array to be reduced.
	    init_value (ArrayType): The initial value for the reduction.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.reduce operation.
	"""
	if isinstance(operand, Array8Lt):
		operand = operand.materialize()
	if isinstance(init_value, Array8Lt):
		init_value = init_value.materialize()
	return lax.reduce(operand, init_value, *args, **kwargs)


@core.primitive_handler("mul")
def handle_mul(
	primitive,
	x: ArrayType,
	y: ArrayType,
):
	"""
	Custom handler for JAX's mul operation.

	Materializes Array8Lt inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): First array to multiply.
	    y (ArrayType): Second array to multiply.

	Returns:
	    The result of lax.mul operation.
	"""
	if isinstance(x, Array8Lt):
		x = x.materialize()
	if isinstance(y, Array8Lt):
		y = y.materialize()
	return lax.mul(x, y)


@core.primitive_handler("transpose")
def handle_transpose(
	primitive,
	operand: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's transpose operation.

	Materializes Array8Lt input before performing the operation.
	Re-quantizes the result if the input was Array8Lt.

	Args:
	    primitive: The JAX primitive being handled.
	    operand (ArrayType): The array to be transposed.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.transpose operation, potentially re-quantized.
	"""
	original_quantized = False
	if isinstance(operand, Array8Lt):
		blocksize = operand.blocksize
		operand = operand.materialize()
		original_quantized = True
	operand = lax.transpose(operand, *args, **kwargs)
	if original_quantized:
		operand = Array8Lt.quantize(
			operand,
			blocksize=blocksize,
			dtype=operand.dtype,
		)
	return operand


@core.primitive_handler("conv_general_dilated")
def handle_conv(
	primitive,
	lhs: ArrayType,
	rhs: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's conv_general_dilated operation.

	Materializes Array8Lt inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    lhs (ArrayType): Left-hand side array (input).
	    rhs (ArrayType): Right-hand side array (kernel).
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.conv operation.
	"""
	if isinstance(lhs, Array8Lt):
		lhs = lhs.materialize()
	if isinstance(rhs, Array8Lt):
		rhs = rhs.materialize()
	return lax.conv_general_dilated(lhs, rhs, *args, **kwargs)


@core.primitive_handler("max")
def handle_max(
	primitive,
	x: ArrayType,
	y: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's max operation.

	Materializes Array8Lt inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): First array for max comparison.
	    y (ArrayType): Second array for max comparison.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.max operation.
	"""
	if isinstance(x, Array8Lt):
		x = x.materialize()
	if isinstance(y, Array8Lt):
		y = y.materialize()
	return lax.max(x, y, *args, **kwargs)


@core.primitive_handler("exp")
def handle_exp(
	primitive,
	x: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's exp operation.

	Materializes Array8Lt input before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): The array to apply exponential to.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.exp operation.
	"""
	if isinstance(x, Array8Lt):
		x = x.materialize()
	return lax.exp(x, *args, **kwargs)


@core.primitive_handler("log")
def handle_log(
	primitive,
	x: ArrayType,
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's log operation.

	Materializes Array8Lt input before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): The array to apply logarithm to.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.log operation.
	"""
	if isinstance(x, Array8Lt):
		x = x.materialize()
	return lax.log(x, *args, **kwargs)


@core.primitive_handler("reshape")
def handle_reshape(
	primitive: Primitive,
	operand: ArrayType,
	**kwargs: Any,
):
	"""
	Custom handler for JAX's reshape operation.

	This function handles reshaping for both regular arrays and Array8Lt quantized arrays.
	It materializes Array8Bit input before reshaping and re-quantizes the result if the input was Array8Bit.

	Args:
	    primitive (Primitive): The JAX primitive being handled.
	    operand (ArrayType): The array to be reshaped.
	    new_sizes (Tuple[int, ...]): The desired new shape of the array.
	    dimensions (Tuple[int, ...], optional): The order in which dimensions should be permuted before reshaping.
	    **kwargs: Additional keyword arguments for the reshape operation.

	Returns:
	    ArrayType: The reshaped array, potentially re-quantized if the input was Array8Lt.

	Raises:
	    ValueError: If the new shape is not compatible with the original array's size.
	"""
	original_quantized = isinstance(operand, Array8Lt)

	if original_quantized:
		blocksize = operand.blocksize
		operand = operand.materialize()

	try:
		reshaped = lax.reshape(operand, **kwargs)
	except ValueError as e:
		raise ValueError(
			f"Reshape operation failed: {str(e)}. "
			f"Ensure the new shape {kwargs} is compatible with the original array size."
		) from e
	if original_quantized:
		reshaped = Array8Lt.quantize(
			operand,
			blocksize=blocksize,
			dtype=operand.dtype,
		)
	return reshaped


@core.primitive_handler("concatenate")
def handle_concatenate(
	primitive,
	operands: Sequence[ArrayType],
	*args,
	**kwargs,
):
	"""
	Custom handler for JAX's concatenate operation.

	Materializes Array8Lt inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    operands (Sequence[ArrayType]): Sequence of arrays to concatenate.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.concatenate operation.
	"""
	materialized_operands = [
		op.materialize() if isinstance(op, Array8Lt) else op for op in operands
	]
	return lax.concatenate(materialized_operands, *args, **kwargs)


@core.primitive_handler("convert_element_type")
def convert_element_type(
	primitive,
	arg: ArrayType,
	**params,
) -> ArrayType:
	"""Handle element type conversion for Array8Lt."""
	result = jax.tree_util.tree_map(
		functools.partial(core.default_handler, primitive, **params), arg
	)
	result.dtype = params["new_dtype"]
	return result


@core.primitive_handler("broadcast_in_dim")
def handle_broadcast_in_dim(
	primitive,
	operand: ArrayType,
	*args,
	**kwargs,
) -> ArrayType:
	"""Handle broadcast_in_dim for Array8Lt."""
	original_quantized = isinstance(operand, Array8Lt)
	array = operand
	if original_quantized:
		blocksize = operand.blocksize
		array = operand.materialize()
	result = jax.lax.broadcast_in_dim(array, *args, **kwargs)
	if original_quantized:
		result = Array8Lt.quantize(
			result,
			blocksize=blocksize,
			dtype=operand.dtype,
		)
	return result


@core.primitive_handler("gather")
def handle_gather(
	primitive,
	operand: ArrayType,
	*args,
	**kwargs,
) -> ArrayType:
	"""Handle gather for Array8Lt."""
	original_quantized = isinstance(operand, Array8Lt)
	array = operand
	if original_quantized:
		array = operand.materialize()
	result = jax.lax.gather(array, *args, **kwargs)
	return result
