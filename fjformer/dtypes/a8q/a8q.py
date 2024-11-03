from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence, Union

import jax
from jax import Array, lax
from jax import numpy as jnp
from jax.core import Primitive

import fjformer.core as core 

@jax.jit
def quantize_row_q8_0(array):
	scale = jnp.max(jnp.abs(array), axis=-1, keepdims=True)
	array = jnp.int8(jnp.rint(array * (127.5 / scale)))
	return array, scale


@jax.jit
def dequantize_row_q8_0(quants, scales):
	dequantized = quants.astype(jnp.float32) * (scales / 127.5)
	return dequantized


@dataclass
class A8Q(core.ImplicitArray):
	"""
	Custom 8-bit Quantized Array for efficient manipulation of JAX arrays.

	This class provides methods for quantizing and dequantizing JAX arrays to 8-bit
	representation, which can significantly reduce memory usage and potentially
	improve computation speed for certain operations.

	Attributes:
	    array_quantized (core.ArrayValue): The quantized array data.
	    scale (core.ArrayValue): Scaling factors for dequantization.
	    shape (tuple): Shape of the quantized array.
	    dtype (jnp.dtype): Original dtype of the array before quantization.

	Example:
	    >>> import jax
	    >>> import jax.numpy as jnp

	    >>> x = jax.random.normal(jax.random.key(0), (512, 64), dtype=jnp.float32)
	    >>> xp = jax.random.normal(jax.random.key(1), (64, 256), dtype=jnp.float32)

	    >>> quantized_x = A8Q.quantize(x)
	    >>> quantized_xp = A8Q.quantize(xp)

	    >>> @jax.jit
	    >>> @core.implicit_compact
	    >>> def f(a, b):
	    ...   return jnp.dot(a, b)

	    >>> result = f(x, xp)
	    >>> q_result = f(quantized_x, quantized_xp)

	    >>> print(jnp.allclose(result, q_result, rtol=1e-2, atol=1e-2))
	    True
	"""

	array_quantized: core.ArrayValue
	scale: core.ArrayValue

	def materialize(self) -> Array:
		"""
		Materialize the quantized array back to its original representation.

		Returns:
		    Array: The dequantized array in its original dtype.
		"""
		return self.dequantize(
			array_quantized=self.array_quantized,
			scale=self.scale,
			float_dtype=self.dtype,
			shape=self.shape,
		)

	@classmethod
	def quantize(
		cls,
		array: Array,
		dtype: Optional[jnp.dtype] = None,
		q8: int = 128,
		*_,
		**__,
	) -> "A8Q":
		"""
		Quantize a JAX array to 8-bit representation.

		Args:
		    array (Array): The input array to quantize.
		    dtype (jnp.dtype, optional): The desired dtype for the output. If None, uses the input array's dtype.

		Returns:
		    A8Q: The quantized array.
		"""

		org_shape = array.shape
		q8 = min(q8, array.size)
		if q8 % array.size != 0:
			q8 = array.shape[-1]
		array = array.reshape(-1, q8)
		quants, scales = quantize_row_q8_0(array=array)

		return cls(
			array_quantized=quants,
			scale=scales,
			shape=org_shape,
			dtype=dtype or array.dtype,
		)

	@staticmethod
	def dequantize(
		array_quantized: Array,
		scale: Array,
		float_dtype: jnp.dtype,
		shape: Sequence[int],
	) -> Array:
		"""
		Dequantize an 8-bit array back to its original representation.

		Args:
		    array_quantized (Array): The quantized array data.
		    scale (Array): The scaling factors used in quantization.
		    float_dtype (jnp.dtype): The desired output dtype.
				shape (Sequence[int]): org array shape
		Returns:
		    Array: The dequantized array.
		"""

		array = (
			dequantize_row_q8_0(
				quants=array_quantized,
				scales=scale,
			)
			.reshape(shape)
			.astype(float_dtype)
		)

		return array

	def __repr__(self) -> str:
		return f"A8Q(quants={self.array_quantized}, shape={self.shape}, dtype={self.dtype})"

	@property
	def nbytes(self) -> int:
		"""
		Calculate the total number of bytes used by the quantized representation.

		Returns:
		    int: The number of bytes used.
		"""
		return self.array_quantized.nbytes + self.scale.nbytes

	def memory_savings(self) -> float:
		"""
		Calculate the memory savings compared to the original array.

		Returns:
		    float: The percentage of memory saved.
		"""
		original_size = jnp.prod(jnp.array(self.shape)) * jnp.dtype(self.dtype).itemsize
		return (1 - self.nbytes / original_size) * 100


ArrayType = Union[Array, A8Q]


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

	Materializes A8Q inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    lhs (ArrayType): Left-hand side array.
	    rhs (ArrayType): Right-hand side array.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.dot_general operation.
	"""
	if isinstance(lhs, A8Q):
		lhs = lhs.materialize()
	if isinstance(rhs, A8Q):
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

	Materializes A8Q inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): First array to add.
	    y (ArrayType): Second array to add.

	Returns:
	    The result of lax.add operation.
	"""
	if isinstance(x, A8Q):
		x = x.materialize()
	if isinstance(y, A8Q):
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

	Materializes A8Q inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    operand (ArrayType): The array to be reduced.
	    init_value (ArrayType): The initial value for the reduction.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.reduce operation.
	"""
	if isinstance(operand, A8Q):
		operand = operand.materialize()
	if isinstance(init_value, A8Q):
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

	Materializes A8Q inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): First array to multiply.
	    y (ArrayType): Second array to multiply.

	Returns:
	    The result of lax.mul operation.
	"""
	if isinstance(x, A8Q):
		x = x.materialize()
	if isinstance(y, A8Q):
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

	Materializes A8Q input before performing the operation.
	Re-quantizes the result if the input was A8Q.

	Args:
	    primitive: The JAX primitive being handled.
	    operand (ArrayType): The array to be transposed.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.transpose operation, potentially re-quantized.
	"""
	original_quantized = False
	if isinstance(operand, A8Q):
		operand = operand.materialize()
		original_quantized = True
	operand = lax.transpose(operand, *args, **kwargs)
	if original_quantized:
		operand = A8Q.quantize(operand, dtype=operand.dtype)
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

	Materializes A8Q inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    lhs (ArrayType): Left-hand side array (input).
	    rhs (ArrayType): Right-hand side array (kernel).
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.conv operation.
	"""
	if isinstance(lhs, A8Q):
		lhs = lhs.materialize()
	if isinstance(rhs, A8Q):
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

	Materializes A8Q inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): First array for max comparison.
	    y (ArrayType): Second array for max comparison.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.max operation.
	"""
	if isinstance(x, A8Q):
		x = x.materialize()
	if isinstance(y, A8Q):
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

	Materializes A8Q input before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): The array to apply exponential to.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.exp operation.
	"""
	if isinstance(x, A8Q):
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

	Materializes A8Q input before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): The array to apply logarithm to.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.log operation.
	"""
	if isinstance(x, A8Q):
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

	This function handles reshaping for both regular arrays and A8Q quantized arrays.
	It materializes ArrayNF4 input before reshaping and re-quantizes the result if the input was ArrayNF4.

	Args:
	    primitive (Primitive): The JAX primitive being handled.
	    operand (ArrayType): The array to be reshaped.
	    new_sizes (Tuple[int, ...]): The desired new shape of the array.
	    dimensions (Tuple[int, ...], optional): The order in which dimensions should be permuted before reshaping.
	    **kwargs: Additional keyword arguments for the reshape operation.

	Returns:
	    ArrayType: The reshaped array, potentially re-quantized if the input was A8Q.

	Raises:
	    ValueError: If the new shape is not compatible with the original array's size.
	"""
	original_quantized = isinstance(operand, A8Q)

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
		reshaped = A8Q.quantize(reshaped, dtype=reshaped.dtype)
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

	Materializes A8Q inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    operands (Sequence[ArrayType]): Sequence of arrays to concatenate.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.concatenate operation.
	"""
	materialized_operands = [
		op.materialize() if isinstance(op, A8Q) else op for op in operands
	]
	return lax.concatenate(materialized_operands, *args, **kwargs)


@core.primitive_handler("convert_element_type")
def convert_element_type(
	primitive,
	arg: ArrayType,
	**params,
) -> ArrayType:
	"""Handle element type conversion for A8Q."""
	result = jax.tree_util.tree_map(
		partial(core.default_handler, primitive, **params), arg
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
	"""Handle broadcast_in_dim for A8Q."""
	original_quantized = isinstance(operand, A8Q)
	array = operand
	if original_quantized:
		array = operand.materialize()
	result = jax.lax.broadcast_in_dim(array, *args, **kwargs)
	if original_quantized:
		result = A8Q.quantize(result, dtype=operand.dtype)
	return result


@core.primitive_handler("gather")
def handle_gather(
	primitive,
	operand: ArrayType,
	*args,
	**kwargs,
) -> ArrayType:
	"""Handle gather for A8Q."""
	original_quantized = isinstance(operand, A8Q)
	array = operand
	if original_quantized:
		array = operand.materialize()
	result = jax.lax.gather(array, *args, **kwargs)
	return result
