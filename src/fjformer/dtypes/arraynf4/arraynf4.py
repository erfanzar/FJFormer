# NF4 impl from QLoRA Paper

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence, Tuple, Union

import chex
import jax
from jax import Array, lax
from jax import numpy as jnp
from jax.core import Primitive
from jax.sharding import NamedSharding, SingleDeviceSharding

import fjformer.core as core
from fjformer.sharding import auto_shard_array, with_sharding_constraint

XB = 2048
SB = 1024
CHUNK_SIZE = 1024**2
NF4 = jnp.array(  # I took this from QLoRA Paper.
	[
		-1.0,
		-0.6961928009986877,
		-0.5250730514526367,
		-0.39491748809814453,
		-0.28444138169288635,
		-0.18477343022823334,
		-0.09105003625154495,
		0.0,
		0.07958029955625534,
		0.16093020141124725,
		0.24611230194568634,
		0.33791524171829224,
		0.44070982933044434,
		0.5626170039176941,
		0.7229568362236023,
		1.0,
	],
	dtype=jnp.float32,
)


def absmax(x, b):
	numel = x.size
	assert numel % b == 0
	assert x.ndim == 1
	nb = numel // b
	return jnp.max(jnp.abs(x.reshape(nb, b)), axis=1)


@jax.jit
def quantize(array: jax.Array):
	xf = array.ravel()
	x_block_u = XB if xf.size % XB == 0 else xf.size
	nXB = xf.size // x_block_u
	XBa = xf.reshape(nXB, x_block_u)
	scalers = absmax(xf, x_block_u)
	scaler_mean = jnp.mean(scalers)
	scaler_1 = scalers - scaler_mean
	sb_block_u = SB if scaler_1.size % SB == 0 else scaler_1.size
	nSB = scaler_1.size // sb_block_u
	scaler_blocks = scaler_1.reshape(nSB, sb_block_u)
	scaler_absmax = absmax(scaler_1, sb_block_u)
	quant_factor = 256 / (2 * scaler_absmax)
	quant_scale = (
		jnp.clip(jnp.round(scaler_blocks * quant_factor[..., None]), min=-128, max=127)
		.flatten()
		.astype(jnp.int8)
	)
	scaled_blocks = XBa / scalers[..., None]
	quant_block = jnp.empty(xf.size, dtype=jnp.uint8)
	flattened = scaled_blocks.ravel()

	def _quantize_chunk(chunk):
		return jnp.argmin(jnp.abs(chunk - NF4), axis=-1).astype(jnp.uint8)

	def _body(cidx, quant_block):
		return quant_block.at[cidx].set(_quantize_chunk(flattened[cidx]))

	quant_block = jax.lax.fori_loop(0, flattened.size, _body, quant_block)
	quant_block = quant_block[::2] << 4 | quant_block[1::2]
	return quant_block.astype(jnp.uint8), quant_scale, quant_factor, scaler_mean


@jax.jit
def dequantize(
	quant_block,
	quant_scale,
	quant_factor,
	scaler_mean,
):
	first_element = quant_block >> 4
	second_element = quant_block & 0b1111
	deq_first = NF4[first_element]
	deq_second = NF4[second_element]
	sb_block_u = SB if quant_scale.size % SB == 0 else quant_scale.size
	nSB = quant_scale.size // sb_block_u
	quant_scale = quant_scale.reshape(nSB, sb_block_u)
	scalers = ((quant_scale / quant_factor[..., None]) + scaler_mean).ravel()
	scalers = scalers.repeat(deq_first.size // scalers.size)
	deq_first.size // scalers.size
	scaled_fir = (deq_first * scalers)[:, None].transpose(1, 0)
	scaled_sec = (deq_second * scalers)[:, None].transpose(1, 0)
	return jnp.stack([scaled_fir, scaled_sec], axis=-1)


@dataclass
class ArrayNF4(core.ImplicitArray):
	quant_block: core.ArrayValue
	quant_scale: core.ArrayValue
	quant_factor: core.ArrayValue
	scaler_mean: core.ArrayValue
	sharding: Union[NamedSharding, SingleDeviceSharding] = core.aux_field()

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

		dequantized_array = (
			dequantize(
				quant_block=self.quant_block,
				quant_scale=self.quant_scale,
				quant_factor=self.quant_factor,
				scaler_mean=self.scaler_mean,
			)
			.reshape(self.shape)
			.astype(dtype or self.dtype)
		)
		if isinstance(self.sharding, NamedSharding):
			with self.sharding.mesh:
				dequantized_array = with_sharding_constraint(
					dequantized_array,
					self.sharding.spec,
				)
		elif isinstance(self.sharding, SingleDeviceSharding):
			dequantized_array = jax.device_put(dequantized_array, self.sharding)
		else:
			raise NotImplementedError(f"Unknown device sharding {self.sharding}")
		return dequantized_array

	@classmethod
	def quantize(cls, array: chex.Array) -> "ArrayNF4":
		sharding = array.sharding
		(
			quant_block,
			quant_scale,
			quant_factor,
			scaler_mean,
		) = quantize(array=array)
		if isinstance(sharding, NamedSharding):
			names = [s for s in sharding.spec if s is not None]
			with sharding.mesh:
				quant_block = auto_shard_array(quant_block, names=names)
				quant_scale = auto_shard_array(quant_scale, names=names)
				quant_factor = auto_shard_array(quant_factor, names=names)
				scaler_mean = auto_shard_array(scaler_mean, names=names)

		elif isinstance(sharding, SingleDeviceSharding):
			# just simply put them on the same device as org array
			quant_block = jax.device_put(quant_block, device=sharding)
			quant_scale = jax.device_put(quant_scale, device=sharding)
			quant_factor = jax.device_put(scaler_mean, device=sharding)
			scaler_mean = jax.device_put(scaler_mean, device=sharding)
		else:
			raise NotImplementedError(f"Unknown device sharding {sharding}")
		return cls(
			quant_block=quant_block,
			quant_scale=quant_scale,
			quant_factor=quant_factor,
			scaler_mean=scaler_mean,
			dtype=array.dtype,
			shape=array.shape,
			sharding=sharding,
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
