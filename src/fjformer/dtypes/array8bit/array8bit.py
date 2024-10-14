# TODO : Implement Custom Backward Prp
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence, Union

import jax
import triton
from jax import Array, lax
from jax import numpy as jnp
from jax.core import Primitive
from triton import language as tl

import fjformer.core as core
from fjformer.jax_triton import strides_from_shape, triton_call, cdiv


@triton.autotune(
	[
		triton.Config({}, num_warps=8, num_stages=2),
		triton.Config({}, num_warps=4, num_stages=2),
		triton.Config({}, num_warps=2, num_stages=2),
		triton.Config({}, num_warps=1, num_stages=2),
	],
	key=["BLOCK_SIZE_K"],
)
@triton.jit
def quantize_row_q8_triton(
	A,
	M,
	K,
	stride_am,
	stride_ak,
	stride_qm,
	stride_qk,
	stride_sm,
	stride_sk,
	Q,
	S,
	BLOCK_SIZE_M: tl.constexpr,
	BLOCK_SIZE_K: tl.constexpr,
):
	pid_m = tl.program_id(axis=0)
	offs_k = tl.arange(0, BLOCK_SIZE_K)
	offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

	a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
	q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

	scales = tl.zeros([BLOCK_SIZE_M, 1], tl.float32) - float("inf")
	for pid_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
		a = tl.load(
			a_ptrs,
			mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - pid_k * BLOCK_SIZE_K),
			other=0.0,
		)
		scales = tl.maximum(scales, tl.max(tl.abs(a)))
		a_ptrs += pid_k * BLOCK_SIZE_K
	a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
	scales = scales / 127.0
	ids = tl.where(scales > 0, 1 / scales, 0)
	for pid_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
		a = tl.load(
			a_ptrs,
			mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - pid_k * BLOCK_SIZE_K),
			other=0.0,
		)
		quant = tl.floor((a * ids) + 0.5).to(tl.int8)
		tl.store(
			q_ptrs,
			quant,
			mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - pid_k * BLOCK_SIZE_K),
		)
		a_ptrs += pid_k * BLOCK_SIZE_K
		q_ptrs += pid_k * BLOCK_SIZE_K

	s_ptrs = S + (offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk)
	tl.store(s_ptrs, scales, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))


def quantize_row_q8_triton_call(array):
	assert array.ndim == 2
	M, K = array.shape
	BLOCK_SIZE_M = 1
	BLOCK_SIZE_K = triton.next_power_of_2(K)

	quants_shape = jax.ShapeDtypeStruct((M, K), jnp.int8)
	scales_shape = jax.ShapeDtypeStruct((M, 1), jnp.float32)

	stride_am, stride_ak = strides_from_shape(array.shape)
	stride_qm, stride_qk = strides_from_shape(quants_shape.shape)
	stride_sm, stride_sk = strides_from_shape(scales_shape.shape)

	quants, scales = triton_call(
		array,
		M,
		K,
		stride_am,
		stride_ak,
		stride_qm,
		stride_qk,
		stride_sm,
		stride_sk,
		kernel=quantize_row_q8_triton,
		BLOCK_SIZE_M=BLOCK_SIZE_M,
		BLOCK_SIZE_K=BLOCK_SIZE_K,
		grid=(cdiv(M, BLOCK_SIZE_M), 1, 1),
		out_shape=[quants_shape, scales_shape],
	)

	return quants, scales.astype(jnp.float16)


@triton.autotune(
	[
		triton.Config({}, num_warps=8, num_stages=2),
		triton.Config({}, num_warps=4, num_stages=2),
		triton.Config({}, num_warps=2, num_stages=2),
		triton.Config({}, num_warps=1, num_stages=2),
	],
	key=["BLOCK_SIZE_K"],
)
@triton.jit
def dequantize_row_q8_triton(
	Q,
	S,
	M,
	K,
	stride_am,
	stride_ak,
	stride_qm,
	stride_qk,
	stride_sm,
	stride_sk,
	A,
	BLOCK_SIZE_M: tl.constexpr,
	BLOCK_SIZE_K: tl.constexpr,
):
	pid_m = tl.program_id(axis=0)
	pid_k = tl.program_id(axis=1)
	offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
	offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

	a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
	q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
	s_ptrs = S + (offs_m[:, None] * stride_sm + offs_k[None, :] * stride_sk)
	quants = tl.load(
		q_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0
	)
	scale = tl.load(
		s_ptrs,
		mask=(offs_m[:, None] < M) & (offs_k[None, :] < 1),
		other=float("-inf"),
	)
	scale = tl.max(scale)
	out = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], tl.float32)
	out += quants * scale
	tl.store(
		a_ptrs,
		out,
		mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
	)


def dequantize_row_q8_triton_call(quants, scales):
	assert quants.ndim == 2
	M, K = quants.shape

	BLOCK_SIZE_M = 1
	BLOCK_SIZE_K = triton.next_power_of_2(K)

	array_shape = jax.ShapeDtypeStruct((M, K), jnp.float32)

	stride_am, stride_ak = strides_from_shape(array_shape.shape)
	stride_qm, stride_qk = strides_from_shape(quants.shape)
	stride_sm, stride_sk = strides_from_shape(scales.shape)

	(array,) = triton_call(
		quants,
		scales,
		M,
		K,
		stride_am,
		stride_ak,
		stride_qm,
		stride_qk,
		stride_sm,
		stride_sk,
		kernel=dequantize_row_q8_triton,
		BLOCK_SIZE_M=BLOCK_SIZE_M,
		BLOCK_SIZE_K=BLOCK_SIZE_K,
		grid=(cdiv(M, BLOCK_SIZE_M), cdiv(K, BLOCK_SIZE_K), 1),
		out_shape=[array_shape],
	)

	return array


@jax.jit
def _mu_quantize_row_q8_0(x):
	"""
	Quantize a row of float32 values to 8-bit integers with blockwise scaling.
	Args:
	    x: input array
	Returns:
	    tuple of (scales, quantized_values)
	    - scales: float16 array of shape (nb,)
	    - quantized_values: int8 array of shape (k,)
	"""
	amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
	d = amax / 127.0
	ids = jnp.where(d > 0, 1.0 / d, 0.0)
	x_scaled = x * ids
	quantized = jnp.round(x_scaled)
	quantized = quantized.astype(jnp.int8)
	return quantized, d.astype(jnp.float16)


@jax.jit
def _mu_dequantize_row_q8_0(quants, scales):
	"""
	Dequantize 8-bit integers back to float32 values using blockwise scaling.

	Args:
	    quants: int8 array of shape (k,) containing quantized values
	    scales: float16 array of shape (nb,) containing scaling factors
	Returns:
	    float32 array of shape (k,) containing dequantized values
	"""
	scales = scales.astype(jnp.float32)
	dequantized = quants * scales
	return dequantized


def quantize_row_q8_0(array):
	match jax.default_backend() == "gpu" and array.size % 16 == 0:
		# case True:
		# 	return quantize_row_q8_triton_call(array)
		case _:
			return _mu_quantize_row_q8_0(array)


def dequantize_row_q8_0(quants, scales):
	match jax.default_backend() == "gpu" and quants.size % 16 == 0:
		# case True:
		# 	return dequantize_row_q8_triton_call(quants, scales)
		case _:
			return _mu_dequantize_row_q8_0(quants, scales)


@dataclass
class Array8Bit(core.ImplicitArray):
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

	    >>> quantized_x = Array8Bit.quantize(x)
	    >>> quantized_xp = Array8Bit.quantize(xp)

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
	) -> "Array8Bit":
		"""
		Quantize a JAX array to 8-bit representation.

		Args:
		    array (Array): The input array to quantize.
		    dtype (jnp.dtype, optional): The desired dtype for the output. If None, uses the input array's dtype.

		Returns:
		    Array8Bit: The quantized array.
		"""
		org_shape = array.shape
		q8 = min(q8, array.size)
		if q8 % array.size != 0:
			q8 = array.shape[-1]
		array = array.reshape(-1, q8)
		quants, scales = quantize_row_q8_0(array)

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
			dequantize_row_q8_0(array_quantized, scale).reshape(shape).astype(float_dtype)
		)

		return array

	def __getitem__(self, idx: Union[int, slice, tuple]) -> Array:
		"""
		Enable indexing of the quantized array.

		Args:
		    idx (Union[int, slice, tuple]): The index or slice to access.

		Returns:
		    Array: The dequantized slice of the array.
		"""
		quantized_slice = self.array_quantized[idx]
		scale_slice = self.scale[idx] if self.scale.ndim > 0 else self.scale
		return self.dequantize(
			quantized_slice,
			scale_slice,
			self.dtype,
			self.shape,
		)

	def __repr__(self) -> str:
		return f"Array8Bit(shape={self.shape}, dtype={self.dtype})"

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


ArrayType = Union[Array, Array8Bit]


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

	Materializes Array8Bit inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    lhs (ArrayType): Left-hand side array.
	    rhs (ArrayType): Right-hand side array.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.dot_general operation.
	"""
	if isinstance(lhs, Array8Bit):
		lhs = lhs.materialize()
	if isinstance(rhs, Array8Bit):
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

	Materializes Array8Bit inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): First array to add.
	    y (ArrayType): Second array to add.

	Returns:
	    The result of lax.add operation.
	"""
	if isinstance(x, Array8Bit):
		x = x.materialize()
	if isinstance(y, Array8Bit):
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

	Materializes Array8Bit inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    operand (ArrayType): The array to be reduced.
	    init_value (ArrayType): The initial value for the reduction.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.reduce operation.
	"""
	if isinstance(operand, Array8Bit):
		operand = operand.materialize()
	if isinstance(init_value, Array8Bit):
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

	Materializes Array8Bit inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): First array to multiply.
	    y (ArrayType): Second array to multiply.

	Returns:
	    The result of lax.mul operation.
	"""
	if isinstance(x, Array8Bit):
		x = x.materialize()
	if isinstance(y, Array8Bit):
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

	Materializes Array8Bit input before performing the operation.
	Re-quantizes the result if the input was Array8Bit.

	Args:
	    primitive: The JAX primitive being handled.
	    operand (ArrayType): The array to be transposed.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.transpose operation, potentially re-quantized.
	"""
	original_quantized = False
	if isinstance(operand, Array8Bit):
		operand = operand.materialize()
		original_quantized = True
	operand = lax.transpose(operand, *args, **kwargs)
	if original_quantized:
		operand = Array8Bit.quantize(operand, dtype=operand.dtype)
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

	Materializes Array8Bit inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    lhs (ArrayType): Left-hand side array (input).
	    rhs (ArrayType): Right-hand side array (kernel).
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.conv operation.
	"""
	if isinstance(lhs, Array8Bit):
		lhs = lhs.materialize()
	if isinstance(rhs, Array8Bit):
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

	Materializes Array8Bit inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): First array for max comparison.
	    y (ArrayType): Second array for max comparison.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.max operation.
	"""
	if isinstance(x, Array8Bit):
		x = x.materialize()
	if isinstance(y, Array8Bit):
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

	Materializes Array8Bit input before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): The array to apply exponential to.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.exp operation.
	"""
	if isinstance(x, Array8Bit):
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

	Materializes Array8Bit input before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    x (ArrayType): The array to apply logarithm to.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.log operation.
	"""
	if isinstance(x, Array8Bit):
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

	This function handles reshaping for both regular arrays and Array8Bit quantized arrays.
	It materializes ArrayNF4 input before reshaping and re-quantizes the result if the input was ArrayNF4.

	Args:
	    primitive (Primitive): The JAX primitive being handled.
	    operand (ArrayType): The array to be reshaped.
	    new_sizes (Tuple[int, ...]): The desired new shape of the array.
	    dimensions (Tuple[int, ...], optional): The order in which dimensions should be permuted before reshaping.
	    **kwargs: Additional keyword arguments for the reshape operation.

	Returns:
	    ArrayType: The reshaped array, potentially re-quantized if the input was Array8Bit.

	Raises:
	    ValueError: If the new shape is not compatible with the original array's size.
	"""
	original_quantized = isinstance(operand, Array8Bit)

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
		reshaped = Array8Bit.quantize(reshaped, dtype=reshaped.dtype)
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

	Materializes Array8Bit inputs before performing the operation.

	Args:
	    primitive: The JAX primitive being handled.
	    operands (Sequence[ArrayType]): Sequence of arrays to concatenate.
	    *args: Variable length argument list.
	    **kwargs: Arbitrary keyword arguments.

	Returns:
	    The result of lax.concatenate operation.
	"""
	materialized_operands = [
		op.materialize() if isinstance(op, Array8Bit) else op for op in operands
	]
	return lax.concatenate(materialized_operands, *args, **kwargs)


@core.primitive_handler("convert_element_type")
def convert_element_type(
	primitive,
	arg: ArrayType,
	**params,
) -> ArrayType:
	"""Handle element type conversion for Array8Bit."""
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
	"""Handle broadcast_in_dim for Array8Bit."""
	original_quantized = isinstance(operand, Array8Bit)
	array = operand
	if original_quantized:
		array = operand.materialize()
	result = jax.lax.broadcast_in_dim(array, *args, **kwargs)
	if original_quantized:
		result = Array8Bit.quantize(result, dtype=operand.dtype)
	return result


@core.primitive_handler("gather")
def handle_gather(
	primitive,
	operand: ArrayType,
	*args,
	**kwargs,
) -> ArrayType:
	"""Handle gather for Array8Bit."""
	original_quantized = isinstance(operand, Array8Bit)
	array = operand
	if original_quantized:
		array = operand.materialize()
	result = jax.lax.gather(array, *args, **kwargs)
	return result
