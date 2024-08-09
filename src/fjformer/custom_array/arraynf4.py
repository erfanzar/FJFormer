# https://arxiv.org/abs/2306.06965 NF4 Isn't Information Theoretically Optimal (and that's Good) Davis Yoshidafrom
# TODO : Implement Custom Backward Prp
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any, Literal, Optional, Sequence, Union

import jax
from jax import Array, lax
from jax import numpy as jnp
from jax.core import Primitive

import fjformer.core as core

NF4 = jnp.array(
    [
        -1.0000,
        -0.6962,
        -0.5251,
        -0.3949,
        -0.2844,
        -0.1848,
        -0.0911,
        0.0000,
        0.0796,
        0.1609,
        0.2461,
        0.3379,
        0.4407,
        0.5626,
        0.7230,
        1.0000,
    ],
    dtype=jnp.float32,
)


def _put_axis_last(array: Array, axis: int) -> Array:
    """
    Transpose the array to put the specified axis last.

    Args:
        array (Array): Input array.
        axis (int): Axis to be moved to the end.

    Returns:
        Array: Transposed array.
    """
    return array.transpose(*chain(range(axis), range(axis + 1, array.ndim), [axis]))


def _pack(array: Array) -> Array:
    """
    Pack two 4-bit values into one 8-bit value.

    Args:
        array (Array): 1D array of 4-bit values.

    Returns:
        Array: Packed array of 8-bit values.

    Raises:
        ValueError: If input array is not 1-dimensional.
    """
    if array.ndim != 1:
        raise ValueError(f"Expected 1D array, got {array.ndim}D array")
    return array[::2] << 4 | array[1::2]


def _unpack(array: Array) -> Array:
    """
    Unpack 8-bit values into two 4-bit values.

    Args:
        array (Array): Array of 8-bit values.

    Returns:
        Array: Unpacked array of 4-bit values.
    """
    return jnp.stack([array >> 4, array & 0xF], axis=-1).reshape(-1)


@partial(
    jax.jit,
    static_argnums=[
        1,
        2,
    ],
)
def _quantize(array, contraction_axis, block_size, factors):

    org_shape = array.shape
    dtype = array.dtype
    transposed = _put_axis_last(array, contraction_axis)

    grouped = transposed.reshape(-1, block_size)
    absmaxes = jnp.max(
        jnp.abs(grouped),
        axis=1,
        keepdims=True,
    )

    scaled = grouped / absmaxes

    assert scaled.ndim == 2
    code_vals = jnp.argmin(jnp.abs(scaled[..., None] - factors), axis=-1).astype(
        jnp.uint8
    )

    array_int = _pack(code_vals.reshape(-1))
    return ArrayNF4(
        absmaxes=absmaxes,
        array_int=array_int,
        block_size=block_size,
        contraction_axis=contraction_axis,
        dtype=dtype,
        shape=org_shape,
    )


# @partial(
#     jax.jit,
#     static_argnums=[
#         5,
#     ],
# )
def _dequantize(
    array_int,
    dtype,
    shape,
    absmaxes,
    contraction_axis,
    block_size,
    factors,
):
    decoded = factors[_unpack(array_int)].reshape(-1, block_size).astype(dtype)

    transposed_shape = (
        shape[:contraction_axis]
        + shape[contraction_axis + 1 :]
        + (shape[contraction_axis],)
    )

    transposed = (decoded * absmaxes).reshape(transposed_shape)

    untranspose = chain(
        range(contraction_axis),
        [transposed.ndim - 1],
        range(contraction_axis, transposed.ndim - 1),
    )

    return transposed.transpose(*untranspose)


@dataclass
class ArrayNF4(core.ImplicitArray):
    """
    https://arxiv.org/abs/2306.06965 NF4 Isn't Information Theoretically Optimal (and that's Good) Davis Yoshidafrom

    Represents a 4-bit quantized array.

    This class provides methods for quantizing and dequantizing arrays,
    as well as utility methods for packing and unpacking 4-bit values.

    Attributes:
        array_int (core.ArrayValue): Packed 4-bit integer representation of the array.
        absmaxes (core.ArrayValue): Absolute maximum values for each block.
        block_size (int): Size of each quantization block.
        contraction_axis (int): Axis along which contraction is performed.
    """

    array_int: core.ArrayValue
    absmaxes: core.ArrayValue

    block_size: int = core.aux_field()
    contraction_axis: int = core.aux_field()

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
        return _dequantize(
            self.array_int,
            dtype,
            self.shape,
            self.absmaxes,
            self.contraction_axis,
            self.block_size,
            NF4,
        )

    @classmethod
    def quantize(
        cls,
        array: Array,
        block_size: Literal[32, 64, 128, 256, 512, 1024, 2048, 4096] = 64,
        contraction_axis: int = -1,
        dtype: Optional[jnp.dtype] = None,
    ) -> "ArrayNF4":
        """
        Quantize a full-precision array into a 4-bit array.

        Args:
            array (Array): Input array to be quantized.
            block_size (Literal[32, 64, 128, 256, 512, 1024, 2048, 4096]): Size of each quantization block.
            contraction_axis (int): Axis along which contraction is performed.
            dtype (Optional[jnp.dtype]): Desired dtype of the quantized array.

        Returns:
            ArrayNF4: Quantized 4-bit array.

        Raises:
            ValueError: If the array shape is incompatible with the specified block_size and contraction_axis.
        """
        if contraction_axis < 0:
            contraction_axis = contraction_axis + array.ndim
        if dtype is not None:
            array = array.astype(dtype)
        if not (array.shape[contraction_axis] / 16).is_integer():
            raise ValueError(
                f"Array shape {array.shape} with contraction_axis {contraction_axis} is incompatible with 4-bit quantization."
            )
        return _quantize(array, contraction_axis, block_size, NF4)


ArrayType = Union[Array, ArrayNF4]


@core.primitive_handler("dot_general")
def handle_dot_general(
    primitive: Primitive,
    lhs: ArrayType,
    rhs: ArrayType,
    *args,
    **kwargs,
):
    """
    Custom handler for JAX's dot_general operation.

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        lhs (ArrayType): Left-hand side array.
        rhs (ArrayType): Right-hand side array.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.dot_general operation.
    """
    if isinstance(lhs, ArrayNF4):
        lhs = lhs.materialize()
    if isinstance(rhs, ArrayNF4):
        rhs = rhs.materialize()
    return lax.dot_general(lhs=lhs, rhs=rhs, *args, **kwargs)


@core.primitive_handler("add")
def handle_add(primitive: Primitive, x: ArrayType, y: ArrayType):
    """
    Custom handler for JAX's add operation.

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        x (ArrayType): First array to add.
        y (ArrayType): Second array to add.

    Returns:
        The result of lax.add operation.
    """
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    if isinstance(y, ArrayNF4):
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

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        operand (ArrayType): The array to be reduced.
        init_value (ArrayType): The initial value for the reduction.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.reduce operation.
    """
    if isinstance(operand, ArrayNF4):
        operand = operand.materialize()
    if isinstance(init_value, ArrayNF4):
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

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        x (ArrayType): First array to multiply.
        y (ArrayType): Second array to multiply.

    Returns:
        The result of lax.mul operation.
    """
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    if isinstance(y, ArrayNF4):
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

    Materializes ArrayNF4 input before performing the operation.
    Re-quantizes the result if the input was ArrayNF4.

    Args:
        primitive: The JAX primitive being handled.
        operand (ArrayType): The array to be transposed.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.transpose operation, potentially re-quantized.
    """
    original_quantized = False
    if isinstance(operand, ArrayNF4):
        array = operand.materialize()
        original_quantized = True
    else:
        array = operand
    array = lax.transpose(array, *args, **kwargs)
    if original_quantized:
        array = ArrayNF4.quantize(
            array=array,
            block_size=operand.block_size,
            contraction_axis=operand.contraction_axis,
            dtype=operand.dtype,
        )
    return array


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

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        lhs (ArrayType): Left-hand side array (input).
        rhs (ArrayType): Right-hand side array (kernel).
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.conv operation.
    """
    if isinstance(lhs, ArrayNF4):
        lhs = lhs.materialize()
    if isinstance(rhs, ArrayNF4):
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

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        x (ArrayType): First array for max comparison.
        y (ArrayType): Second array for max comparison.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.max operation.
    """
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    if isinstance(y, ArrayNF4):
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

    Materializes ArrayNF4 input before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        x (ArrayType): The array to apply exponential to.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.exp operation.
    """
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    return lax.exp(x, *args, **kwargs)


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
        primitive (Primitive): The JAX primitive being handled.
        x (ArrayType): The array to apply logarithm to.
        **kwargs: Additional keyword arguments for the log operation.

    Returns:
        jnp.ndarray: The result of the natural logarithm operation.

    Raises:
        ValueError: If the input contains non-positive values.

    Note:
        This operation always returns a full-precision array, even if the input
        was an ArrayNF4, as the logarithm operation typically produces non-integer results.
    """
    if isinstance(x, ArrayNF4):
        x = x.materialize()
    try:
        return lax.log(x, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Log operation failed: {str(e)}") from e


@core.primitive_handler("reshape")
def handle_reshape(
    primitive: Primitive,
    operand: ArrayType,
    **kwargs: Any,
):
    """
    Custom handler for JAX's reshape operation.

    This function handles reshaping for both regular arrays and ArrayNF4 quantized arrays.
    It materializes ArrayNF4 input before reshaping and re-quantizes the result if the input was ArrayNF4.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        operand (ArrayType): The array to be reshaped.
        new_sizes (Tuple[int, ...]): The desired new shape of the array.
        dimensions (Tuple[int, ...], optional): The order in which dimensions should be permuted before reshaping.
        **kwargs: Additional keyword arguments for the reshape operation.

    Returns:
        ArrayType: The reshaped array, potentially re-quantized if the input was ArrayNF4.

    Raises:
        ValueError: If the new shape is not compatible with the original array's size.
    """
    original_quantized = isinstance(operand, ArrayNF4)
    if original_quantized:
        array = operand.materialize()
    else:
        array = operand

    start_shape = array.shape
    try:
        reshaped = lax.reshape(array, **kwargs)
    except ValueError as e:
        raise ValueError(
            f"Reshape operation failed: {str(e)}. "
            f"Ensure the new shape {kwargs} is compatible with the original array size."
        ) from e

    if original_quantized:
        q_dim = start_shape[operand.contraction_axis]
        new_idx = [rg for rg, shape_ in enumerate(reshaped.shape) if shape_ == q_dim][0]
        return ArrayNF4.quantize(
            array=reshaped,
            block_size=operand.block_size,
            contraction_axis=new_idx,
            dtype=operand.dtype,
        )
    return reshaped


@core.primitive_handler("concatenate")
def handle_concatenate(
    primitive: Primitive,
    operands: Sequence[ArrayType],
    *args,
    **kwargs,
):
    """
    Custom handler for JAX's concatenate operation.

    Materializes ArrayNF4 inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        operands (Sequence[ArrayType]): Sequence of arrays to concatenate.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.concatenate operation.
    """
    materialized_operands = [
        op.materialize() if isinstance(op, ArrayNF4) else op for op in operands
    ]
    return lax.concatenate(materialized_operands, *args, **kwargs)


@core.primitive_handler("convert_element_type")
def convert_element_type(
    primitive: Primitive,
    arg: ArrayNF4,
    **params,
) -> ArrayNF4:
    """Handle element type conversion for ArrayNF4."""
    result = jax.tree_util.tree_map(
        partial(core.default_handler, primitive, **params), arg
    )
    result.dtype = params["new_dtype"]
    return result


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
        primitive (Any): The JAX primitive being handled.
        sym (core.symbols.SymbolicConstant): The symbolic constant being operated on.
        **kwargs: Additional keyword arguments for the primitive operation.

    Returns:
        core.symbols.SymbolicConstant: A new SymbolicConstant with potentially
        updated shape and dtype, but the same value as the input.
    """
    out_shape, out_dtype = _out_shape_dtype(primitive, sym, **kwargs)
    return core.symbols.SymbolicConstant(sym.value, shape=out_shape, dtype=out_dtype)


def _out_shape_dtype(
    primitive: Any,
    *args: Any,
    **kwargs: Any,
):
    """
    Determine the output shape and dtype for a given primitive operation.

    This function uses JAX's shape inference capabilities to determine what
    the shape and dtype of the output would be if the operation were performed
    on concrete arrays.

    Args:
        primitive (Any): JAX primitive to be evaluated.
        *args: Positional arguments for the primitive.
        **kwargs: Keyword arguments for the primitive.

    Returns:
        Tuple[Tuple[int, ...], Any]: A tuple containing:
            - The shape of the output as a tuple of integers.
            - The dtype of the output.
    """
    out_aval = jax.eval_shape(
        partial(core.default_handler, primitive, **kwargs),
        *(jax.core.get_aval(x) for x in args),
    )
    return jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), out_aval)


@core.primitive_handler("broadcast_in_dim")
def handle_broadcast_in_dim(
    primitive,
    operand: ArrayNF4,
    *args,
    **kwargs,
) -> ArrayNF4:
    """Handle broadcast_in_dim for ArrayNF4."""
    original_quantized = isinstance(operand, ArrayNF4)
    array = operand
    if original_quantized:
        array = operand.materialize()
    result = jax.lax.broadcast_in_dim(array, *args, **kwargs)
    if original_quantized:
        result = ArrayNF4.quantize(
            array=result,
            block_size=operand.block_size,
            contraction_axis=operand.contraction_axis,
            dtype=operand.dtype,
        )
    return result


@core.primitive_handler("gather")
def handle_gather(
    primitive,
    operand: ArrayNF4,
    *args,
    **kwargs,
) -> ArrayNF4:
    """Handle gather for ArrayNF4."""
    original_quantized = isinstance(operand, ArrayNF4)
    array = operand
    if original_quantized:
        array = operand.materialize()
    result = jax.lax.gather(array, *args, **kwargs)
    return result
