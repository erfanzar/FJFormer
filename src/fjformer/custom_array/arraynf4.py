# TODO:Impl QLoRA
# import math
# from typing import Tuple

# import chex
# import jax
# import torch
# from jax import numpy as jnp

# torch.manual_seed(42)
# CHUNK_SIZE = 1024**2
# NF4 = jnp.array(
#     [
#         -1.0,
#         -0.6961928009986877,
#         -0.5250730514526367,
#         -0.39491748809814453,
#         -0.28444138169288635,
#         -0.18477343022823334,
#         -0.09105003625154495,
#         0.0,
#         0.07958029955625534,
#         0.16093020141124725,
#         0.24611230194568634,
#         0.33791524171829224,
#         0.44070982933044434,
#         0.5626170039176941,
#         0.7229568362236023,
#         1.0,
#     ],
#     dtype=jnp.float32,
# )


# def quantize_tensor_nearest(value: chex.Array, nf4: chex.Array) -> chex.Array:
#     value = jnp.expand_dims(value, -1)
#     diff = jnp.abs(value - nf4)
#     closest_nf4 = jnp.argmin(diff, axis=-1)
#     return closest_nf4


# def get_block_absmax(inpt_tensor: chex.Array, block_size: int) -> chex.Array:
#     assert inpt_tensor.ndim == 1
#     assert (inpt_tensor.size % block_size) == 0

#     n_blocks = inpt_tensor.size // block_size
#     blocks = inpt_tensor.reshape(n_blocks, block_size)
#     return jnp.max(jnp.abs(blocks), axis=1)


# def double_quantize_scalers(
#     inpt_tensor: chex.Array,
#     block_size: int,
#     scaler_block_size: int,
# ) -> Tuple[chex.Array, chex.Array, chex.Array]:
#     assert inpt_tensor.ndim == 1
#     assert (inpt_tensor.size % scaler_block_size) == 0

#     scalers_1 = get_block_absmax(inpt_tensor, block_size)
#     scalers_1_mean = jnp.mean(scalers_1)
#     scalers_1 = scalers_1 - scalers_1_mean
#     # Second round of quantization
#     assert scalers_1.size % scaler_block_size == 0
#     n_scaler_blocks = scalers_1.size // scaler_block_size
#     scaler_blocks = scalers_1.reshape(n_scaler_blocks, scaler_block_size)

#     scaler_absmax = get_block_absmax(scalers_1, scaler_block_size)
#     scaler_absmax = jnp.broadcast_to(
#         jnp.expand_dims(scaler_absmax, -1), (n_scaler_blocks, scaler_block_size)
#     )

#     quantization_factor = 256 / (2 * scaler_absmax)
#     quantized_scaler_blocks = scaler_blocks * quantization_factor
#     quantized_scaler_blocks = jnp.round(quantized_scaler_blocks)
#     quantized_scaler_blocks = jnp.clip(quantized_scaler_blocks, min=-128, max=127)

#     quantization_factor = quantization_factor[:, 0]

#     return (
#         quantized_scaler_blocks.flatten().astype(jnp.int8),
#         jax.lax.stop_gradient(quantization_factor.reshape(n_scaler_blocks)),
#         scalers_1_mean,
#     )


# def convert_to_norm_float_weight(
#     inpt_tensor: chex.Array,
#     n_blocks: int,
#     block_size: int,
#     nf4: chex.Array,
# ) -> chex.Array:
#     flattened_tensor = inpt_tensor.flatten()
#     numel = inpt_tensor.size
#     assert numel % 2 == 0
#     blocks = flattened_tensor.reshape(n_blocks, block_size)

#     # Scale the blocks
#     scalers = get_block_absmax(inpt_tensor.flatten(), block_size)
#     scales = jnp.broadcast_to(jnp.expand_dims(scalers, -1), (n_blocks, block_size))
#     scaled_blocks = blocks / scales

#     quantized_blocks = jnp.empty(numel, dtype=jnp.uint8)
#     flattened = scaled_blocks.flatten()

#     for chunk_num in range(math.ceil(numel / CHUNK_SIZE)):
#         start = chunk_num * CHUNK_SIZE
#         end = min(start + CHUNK_SIZE, numel)
#         quantized_blocks = quantized_blocks.at[start:end].set(
#             quantize_tensor_nearest(
#                 flattened[start:end],
#                 nf4,
#             ).astype(jnp.uint8)
#         )

#     combined_blocks = quantized_blocks[::2] << 4 | quantized_blocks[1::2]

#     return combined_blocks.astype(jnp.uint8)


# def dequantize_scalers(
#     quantized_scalers: chex.Array,
#     quantization_factor: chex.Array,
#     scaler_mean: chex.Array,
#     scaler_block_size: int,
#     dtype: jnp.dtype,
# ) -> chex.Array:
#     assert quantized_scalers.ndim == 1
#     assert (quantized_scalers.size % scaler_block_size) == 0
#     n_scaler_blocks = quantized_scalers.size // scaler_block_size
#     quantized_scalers = quantized_scalers.reshape(n_scaler_blocks, scaler_block_size)
#     dequantized = (
#         quantized_scalers / jnp.expand_dims(quantization_factor, -1)
#     ).flatten().astype(dtype) + scaler_mean
#     return dequantized


# def quantize_to_nf4(
#     array: chex.Array,
#     block_size: int,
#     scaler_block_size: int,
# ):
#     assert array.ndim <= 2
#     assert array.size % block_size == 0
#     n_blocks = array.size // block_size

#     (
#         quantized_scalers,
#         quantization_factor,
#         scaler_mean,
#     ) = double_quantize_scalers(
#         array.flatten(),
#         block_size,
#         scaler_block_size,
#     )

#     quantized_data = convert_to_norm_float_weight(
#         array,
#         n_blocks,
#         block_size,
#         NF4,
#     )
#     return quantized_data, quantized_scalers, quantization_factor, scaler_mean


# def dequantize_nf4(
#     quantized_data: chex.Array,
#     quantized_scalers: chex.Array,
#     quantization_factor: chex.Array,
#     scaler_mean: chex.Array,
#     scaler_block_size: chex.Array,
#     block_size: int,
#     org_dtype: jnp.dtype,
#     org_shape: chex.Shape,
# ):

#     first_elements = (quantized_data >> 4).astype("i4")
#     second_elements = (quantized_data & 0b1111).astype("i4")

#     # Dequantize every element
#     dequantized_first = NF4[first_elements]
#     dequantized_second = NF4[second_elements]

#     scalers = dequantize_scalers(
#         quantized_scalers=quantized_scalers,
#         quantization_factor=quantization_factor,
#         scaler_mean=scaler_mean,
#         scaler_block_size=scaler_block_size,
#         dtype=org_dtype,
#     )
#     repeated = jnp.broadcast_to(
#         jnp.expand_dims(scalers, -1),
#         (scalers.shape[0], block_size // 2),
#     )

#     scaled_first = dequantized_first * repeated.flatten()
#     scaled_second = dequantized_second * repeated.flatten()

#     scaled_first = jnp.expand_dims(scaled_first, -1).transpose(1, 0)
#     scaled_second = jnp.expand_dims(scaled_second, -1).transpose(1, 0)
#     return jnp.stack([scaled_first, scaled_second], axis=-1).reshape(org_shape)


# def main():
#     array = torch.randn((1, 64)).cpu().detach().numpy()
#     (
#         quantized_data,
#         quantized_scalers,
#         quantization_factor,
#         scaler_mean,
#     ) = quantize_to_nf4(array, 32, 2)
#     dequantized_array = dequantize_nf4(
#         quantized_data=quantized_data,
#         quantized_scalers=quantized_scalers,
#         quantization_factor=quantization_factor,
#         scaler_mean=scaler_mean,
#         block_size=32,
#         scaler_block_size=2,
#         org_dtype=array.dtype,
#         org_shape=array.shape,
#     )
#     print(array)
#     print(dequantized_array)


# if __name__ == "__main__":
#     main()


# @dataclass
# class ArrayNF4(core.ImplicitArray):

#     array_int: core.ArrayValue
#     absmaxes: core.ArrayValue

#     block_size: int = core.aux_field()
#     contraction_axis: int = core.aux_field()

#     def materialize(self) -> Array:
#         """
#         Materialize the 4-bit array into a full-precision array.

#         Returns:
#             Array: The dequantized array.
#         """
#         return self.dequantize()

#     def dequantize(self, dtype: Optional[jnp.dtype] = None) -> Array:
#         """
#         Dequantize the 4-bit array into a full-precision array.

#         Args:
#             dtype (Optional[jnp.dtype]): Desired dtype of the output array.

#         Returns:
#             Array: Dequantized array.
#         """
#         dtype = dtype if dtype is not None else self.dtype
#         return _dequantize(
#             self.array_int,
#             dtype,
#             self.shape,
#             self.absmaxes,
#             self.contraction_axis,
#             self.block_size,
#             NF4,
#         )

#     @classmethod
#     def quantize(
#         cls,
#         array: Array,
#         block_size: Literal[32, 64, 128, 256, 512, 1024, 2048, 4096] = 64,
#         contraction_axis: int = 0,
#         dtype: Optional[jnp.dtype] = None,
#     ) -> "ArrayNF4":
#         """
#         Quantize a full-precision array into a 4-bit array.

#         Args:
#             array (Array): Input array to be quantized.
#             block_size (Literal[32, 64, 128, 256, 512, 1024, 2048, 4096]): Size of each quantization block.
#             contraction_axis (int): Axis along which contraction is performed.
#             dtype (Optional[jnp.dtype]): Desired dtype of the quantized array.

#         Returns:
#             ArrayNF4: Quantized 4-bit array.

#         Raises:
#             ValueError: If the array shape is incompatible with the specified block_size and contraction_axis.
#         """
#         if contraction_axis < 0:
#             contraction_axis = contraction_axis + array.ndim
#         if dtype is not None:
#             array = array.astype(dtype)
#         if not (array.shape[contraction_axis] / 2).is_integer():
#             raise ValueError(
#                 f"Array shape {array.shape} with contraction_axis {contraction_axis} is incompatible with 4-bit quantization."
#             )
#         return _quantize(array, contraction_axis, block_size, NF4)


# ArrayType = Union[Array, ArrayNF4]


# @core.primitive_handler("dot_general")
# def handle_dot_general(
#     primitive: Primitive,
#     lhs: ArrayType,
#     rhs: ArrayType,
#     *args,
#     **kwargs,
# ):
#     """
#     Custom handler for JAX's dot_general operation.

#     Materializes ArrayNF4 inputs before performing the operation.

#     Args:
#         primitive: The JAX primitive being handled.
#         lhs (ArrayType): Left-hand side array.
#         rhs (ArrayType): Right-hand side array.
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.

#     Returns:
#         The result of lax.dot_general operation.
#     """
#     if isinstance(lhs, ArrayNF4):
#         lhs = lhs.materialize()
#     if isinstance(rhs, ArrayNF4):
#         rhs = rhs.materialize()
#     return lax.dot_general(lhs=lhs, rhs=rhs, *args, **kwargs)


# @core.primitive_handler("add")
# def handle_add(primitive: Primitive, x: ArrayType, y: ArrayType):
#     """
#     Custom handler for JAX's add operation.

#     Materializes ArrayNF4 inputs before performing the operation.

#     Args:
#         primitive: The JAX primitive being handled.
#         x (ArrayType): First array to add.
#         y (ArrayType): Second array to add.

#     Returns:
#         The result of lax.add operation.
#     """
#     if isinstance(x, ArrayNF4):
#         x = x.materialize()
#     if isinstance(y, ArrayNF4):
#         y = y.materialize()
#     return lax.add(x, y)


# @core.primitive_handler("reduce")
# def handle_reduce(
#     primitive,
#     operand: ArrayType,
#     init_value: ArrayType,
#     *args,
#     **kwargs,
# ):
#     """
#     Custom handler for JAX's reduce operation.

#     Materializes ArrayNF4 inputs before performing the operation.

#     Args:
#         primitive: The JAX primitive being handled.
#         operand (ArrayType): The array to be reduced.
#         init_value (ArrayType): The initial value for the reduction.
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.

#     Returns:
#         The result of lax.reduce operation.
#     """
#     if isinstance(operand, ArrayNF4):
#         operand = operand.materialize()
#     if isinstance(init_value, ArrayNF4):
#         init_value = init_value.materialize()
#     return lax.reduce(operand, init_value, *args, **kwargs)


# @core.primitive_handler("mul")
# def handle_mul(
#     primitive,
#     x: ArrayType,
#     y: ArrayType,
# ):
#     """
#     Custom handler for JAX's mul operation.

#     Materializes ArrayNF4 inputs before performing the operation.

#     Args:
#         primitive: The JAX primitive being handled.
#         x (ArrayType): First array to multiply.
#         y (ArrayType): Second array to multiply.

#     Returns:
#         The result of lax.mul operation.
#     """
#     if isinstance(x, ArrayNF4):
#         x = x.materialize()
#     if isinstance(y, ArrayNF4):
#         y = y.materialize()
#     return lax.mul(x, y)


# @core.primitive_handler("transpose")
# def handle_transpose(
#     primitive,
#     operand: ArrayType,
#     *args,
#     **kwargs,
# ):
#     """
#     Custom handler for JAX's transpose operation.

#     Materializes ArrayNF4 input before performing the operation.
#     Re-quantizes the result if the input was ArrayNF4.

#     Args:
#         primitive: The JAX primitive being handled.
#         operand (ArrayType): The array to be transposed.
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.

#     Returns:
#         The result of lax.transpose operation, potentially re-quantized.
#     """
#     original_quantized = False
#     if isinstance(operand, ArrayNF4):
#         array = operand.materialize()
#         original_quantized = True
#     else:
#         array = operand
#     array = lax.transpose(array, *args, **kwargs)
#     if original_quantized:
#         array = ArrayNF4.quantize(
#             array=array,
#             block_size=operand.block_size,
#             contraction_axis=operand.contraction_axis,
#             dtype=operand.dtype,
#         )
#     return array


# @core.primitive_handler("conv_general_dilated")
# def handle_conv(
#     primitive,
#     lhs: ArrayType,
#     rhs: ArrayType,
#     *args,
#     **kwargs,
# ):
#     """
#     Custom handler for JAX's conv_general_dilated operation.

#     Materializes ArrayNF4 inputs before performing the operation.

#     Args:
#         primitive: The JAX primitive being handled.
#         lhs (ArrayType): Left-hand side array (input).
#         rhs (ArrayType): Right-hand side array (kernel).
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.

#     Returns:
#         The result of lax.conv operation.
#     """
#     if isinstance(lhs, ArrayNF4):
#         lhs = lhs.materialize()
#     if isinstance(rhs, ArrayNF4):
#         rhs = rhs.materialize()
#     return lax.conv_general_dilated(lhs, rhs, *args, **kwargs)


# @core.primitive_handler("max")
# def handle_max(
#     primitive,
#     x: ArrayType,
#     y: ArrayType,
#     *args,
#     **kwargs,
# ):
#     """
#     Custom handler for JAX's max operation.

#     Materializes ArrayNF4 inputs before performing the operation.

#     Args:
#         primitive: The JAX primitive being handled.
#         x (ArrayType): First array for max comparison.
#         y (ArrayType): Second array for max comparison.
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.

#     Returns:
#         The result of lax.max operation.
#     """
#     if isinstance(x, ArrayNF4):
#         x = x.materialize()
#     if isinstance(y, ArrayNF4):
#         y = y.materialize()
#     return lax.max(x, y, *args, **kwargs)


# @core.primitive_handler("exp")
# def handle_exp(
#     primitive,
#     x: ArrayType,
#     *args,
#     **kwargs,
# ):
#     """
#     Custom handler for JAX's exp operation.

#     Materializes ArrayNF4 input before performing the operation.

#     Args:
#         primitive: The JAX primitive being handled.
#         x (ArrayType): The array to apply exponential to.
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.

#     Returns:
#         The result of lax.exp operation.
#     """
#     if isinstance(x, ArrayNF4):
#         x = x.materialize()
#     return lax.exp(x, *args, **kwargs)


# @core.primitive_handler("log")
# def handle_log(
#     primitive: Primitive,
#     x: ArrayType,
#     **kwargs: Any,
# ) -> jnp.ndarray:
#     """
#     Custom handler for JAX's log operation.

#     This function computes the natural logarithm of the input, handling both
#     regular arrays and ArrayNF4 quantized arrays.

#     Args:
#         primitive (Primitive): The JAX primitive being handled.
#         x (ArrayType): The array to apply logarithm to.
#         **kwargs: Additional keyword arguments for the log operation.

#     Returns:
#         jnp.ndarray: The result of the natural logarithm operation.

#     Raises:
#         ValueError: If the input contains non-positive values.

#     Note:
#         This operation always returns a full-precision array, even if the input
#         was an ArrayNF4, as the logarithm operation typically produces non-integer results.
#     """
#     if isinstance(x, ArrayNF4):
#         x = x.materialize()
#     try:
#         return lax.log(x, **kwargs)
#     except Exception as e:
#         raise RuntimeError(f"Log operation failed: {str(e)}") from e


# @core.primitive_handler("reshape")
# def handle_reshape(
#     primitive: Primitive,
#     operand: ArrayType,
#     **kwargs: Any,
# ):
#     """
#     Custom handler for JAX's reshape operation.

#     This function handles reshaping for both regular arrays and ArrayNF4 quantized arrays.
#     It materializes ArrayNF4 input before reshaping and re-quantizes the result if the input was ArrayNF4.

#     Args:
#         primitive (Primitive): The JAX primitive being handled.
#         operand (ArrayType): The array to be reshaped.
#         new_sizes (Tuple[int, ...]): The desired new shape of the array.
#         dimensions (Tuple[int, ...], optional): The order in which dimensions should be permuted before reshaping.
#         **kwargs: Additional keyword arguments for the reshape operation.

#     Returns:
#         ArrayType: The reshaped array, potentially re-quantized if the input was ArrayNF4.

#     Raises:
#         ValueError: If the new shape is not compatible with the original array's size.
#     """
#     original_quantized = isinstance(operand, ArrayNF4)
#     if original_quantized:
#         array = operand.materialize()
#     else:
#         array = operand

#     start_shape = array.shape
#     try:
#         reshaped = lax.reshape(array, **kwargs)
#     except ValueError as e:
#         raise ValueError(
#             f"Reshape operation failed: {str(e)}. "
#             f"Ensure the new shape {kwargs} is compatible with the original array size."
#         ) from e

#     if original_quantized:
#         q_dim = start_shape[operand.contraction_axis]
#         new_idx = [rg for rg, shape_ in enumerate(reshaped.shape) if shape_ == q_dim][0]
#         return ArrayNF4.quantize(
#             array=reshaped,
#             block_size=operand.block_size,
#             contraction_axis=new_idx,
#             dtype=operand.dtype,
#         )
#     return reshaped


# @core.primitive_handler("concatenate")
# def handle_concatenate(
#     primitive: Primitive,
#     operands: Sequence[ArrayType],
#     *args,
#     **kwargs,
# ):
#     """
#     Custom handler for JAX's concatenate operation.

#     Materializes ArrayNF4 inputs before performing the operation.

#     Args:
#         primitive: The JAX primitive being handled.
#         operands (Sequence[ArrayType]): Sequence of arrays to concatenate.
#         *args: Variable length argument list.
#         **kwargs: Arbitrary keyword arguments.

#     Returns:
#         The result of lax.concatenate operation.
#     """
#     materialized_operands = [
#         op.materialize() if isinstance(op, ArrayNF4) else op for op in operands
#     ]
#     return lax.concatenate(materialized_operands, *args, **kwargs)


# @core.primitive_handler("convert_element_type")
# def convert_element_type(
#     primitive: Primitive,
#     arg: ArrayNF4,
#     **params,
# ) -> ArrayNF4:
#     """Handle element type conversion for ArrayNF4."""
#     result = jax.tree_util.tree_map(
#         partial(core.default_handler, primitive, **params), arg
#     )
#     result.dtype = params["new_dtype"]
#     return result


# @core.primitive_handler(
#     [
#         "reshape",
#         "broadcast_in_dim",
#         "reduce_min",
#         "reduce_max",
#         "reduce_or",
#         "reduce_and",
#     ]
# )
# def unchanged_value_op(
#     primitive: Any,
#     sym: core.symbols.SymbolicConstant,
#     **kwargs: Any,
# ) -> core.symbols.SymbolicConstant:
#     """
#     Handler for JAX primitives that don't change the value of a SymbolicConstant.

#     This function handles operations that may change the shape or other properties
#     of a SymbolicConstant, but not its fundamental value.

#     Args:
#         primitive (Any): The JAX primitive being handled.
#         sym (core.symbols.SymbolicConstant): The symbolic constant being operated on.
#         **kwargs: Additional keyword arguments for the primitive operation.

#     Returns:
#         core.symbols.SymbolicConstant: A new SymbolicConstant with potentially
#         updated shape and dtype, but the same value as the input.
#     """
#     out_shape, out_dtype = _out_shape_dtype(primitive, sym, **kwargs)
#     return core.symbols.SymbolicConstant(sym.value, shape=out_shape, dtype=out_dtype)


# def _out_shape_dtype(
#     primitive: Any,
#     *args: Any,
#     **kwargs: Any,
# ):
#     """
#     Determine the output shape and dtype for a given primitive operation.

#     This function uses JAX's shape inference capabilities to determine what
#     the shape and dtype of the output would be if the operation were performed
#     on concrete arrays.

#     Args:
#         primitive (Any): JAX primitive to be evaluated.
#         *args: Positional arguments for the primitive.
#         **kwargs: Keyword arguments for the primitive.

#     Returns:
#         Tuple[Tuple[int, ...], Any]: A tuple containing:
#             - The shape of the output as a tuple of integers.
#             - The dtype of the output.
#     """
#     out_aval = jax.eval_shape(
#         partial(core.default_handler, primitive, **kwargs),
#         *(jax.core.get_aval(x) for x in args),
#     )
#     return jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), out_aval)


# @core.primitive_handler("broadcast_in_dim")
# def handle_broadcast_in_dim(
#     primitive,
#     operand: ArrayNF4,
#     *args,
#     **kwargs,
# ) -> ArrayNF4:
#     """Handle broadcast_in_dim for ArrayNF4."""
#     original_quantized = isinstance(operand, ArrayNF4)
#     array = operand
#     if original_quantized:
#         array = operand.materialize()
#     result = jax.lax.broadcast_in_dim(array, *args, **kwargs)
#     if original_quantized:
#         result = ArrayNF4.quantize(
#             array=result,
#             block_size=operand.block_size,
#             contraction_axis=operand.contraction_axis,
#             dtype=operand.dtype,
#         )
#     return result


# @core.primitive_handler("gather")
# def handle_gather(
#     primitive,
#     operand: ArrayNF4,
#     *args,
#     **kwargs,
# ) -> ArrayNF4:
#     """Handle gather for ArrayNF4."""
#     original_quantized = isinstance(operand, ArrayNF4)
#     array = operand
#     if original_quantized:
#         array = operand.materialize()
#     result = jax.lax.gather(array, *args, **kwargs)
#     return result
