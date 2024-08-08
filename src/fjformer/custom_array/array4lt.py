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

# Define a global codebook for quantization
CODE = jnp.linspace(-1.0, 1.0, 256)
BLOCKSIZE = 256


@jax.jit
def quantize_block(A, absmax):
    """
    Quantize the input array with the given absolute maximum values.

    Args:
        A (jnp.ndarray): The input array to be quantized.
        absmax (jnp.ndarray): The absolute maximum values for each element.

    Returns:
        jnp.ndarray: The quantized array.
    """
    reshape_dim = (A.shape[0],)
    if A.ndim > 1:
        reshape_dim += (1,) * (A.ndim - 1)
    normed_values = A / absmax.reshape(reshape_dim)
    indices = jnp.searchsorted(CODE, normed_values, side="left")

    dist_left = jnp.abs(normed_values - CODE[indices])
    dist_right = jnp.abs(
        normed_values - CODE[jnp.clip(indices + 1, 0, CODE.shape[0] - 1)],
    )
    indices = jnp.where(
        dist_right < dist_left,
        jnp.clip(indices + 1, 0, CODE.shape[0] - 1),
        indices,
    )

    return indices.astype(jnp.uint8)


@functools.partial(jax.jit)
def quantize_jax(A):
    """
    Quantize the input array.

    Args:
        A (jnp.ndarray): The input array to be quantized.

    Returns:
        Tuple[jnp.ndarray,jnp.ndarray]: The quantized array, The absolute maximum values for each block.
    """
    n = A.shape[0]
    num_blocks = (n + BLOCKSIZE - 1) // BLOCKSIZE

    indices = jnp.arange(n)
    block_indices = indices // BLOCKSIZE

    absmax_blocks = jnp.zeros((num_blocks,))
    for i in range(num_blocks):
        start = i * BLOCKSIZE
        end = min((i + 1) * BLOCKSIZE, n)
        absmax_blocks = absmax_blocks.at[i].set(jnp.max(jnp.abs(A[start:end])))

    absmax = absmax_blocks[block_indices]

    quantized_A = quantize_block(A, absmax)

    return quantized_A, absmax_blocks


@jax.jit
def dequantize_jax(A, absmax):
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
    block_indices = indices // BLOCKSIZE
    extra_dim = ()
    if CODE[A].ndim > 1:
        extra_dim = (1,) * (len(CODE[A].shape) - 1)
    dequantized_A = CODE[A] * jnp.reshape(
        absmax[block_indices],
        (CODE[A].shape[0],) + extra_dim,
    )
    return dequantized_A


@dataclass
class Array4Lt(core.ImplicitArray):
    """
    Custom 4-bit Quantized Array for efficient manipulation of JAX arrays.

    This class provides methods for quantizing and dequantizing JAX arrays to 4-bit
    representation, which can significantly reduce memory usage and potentially
    improve computation speed for certain operations.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp

        >>> x = jax.random.normal(jax.random.key(0), (512, 64), dtype=jnp.float32)
        >>> xp = jax.random.normal(jax.random.key(1), (64, 256), dtype=jnp.float32)

        >>> quantized_x = Array4Lt.quantize(x, )
        >>> quantized_xp = Array4Lt.quantize(xp, )

        >>> @jax.jit
        >>> @core.implicit_compact
        >>> def f(a, b):
        ...     return jnp.dot(a, b)

        >>> result = f(x, xp)
        >>> q_result = f(quantized_x, quantized_xp)

        >>> print(jnp.allclose(result, q_result, rtol=1e-2, atol=1e-2))
        True
    """

    A: core.ArrayValue
    absmax: core.ArrayValue

    def materialize(self) -> Array:
        """
        Materialize the quantized array back to its original representation.

        Returns:
            Array: The dequantized array in its original dtype.
        """
        return self.dequantize(
            A=self.A,
            absmax=self.absmax,
            float_dtype=self.dtype,
        )

    @classmethod
    def quantize(
        cls,
        array: Array,
        dtype: Optional[jnp.dtype] = None,
    ) -> Array4Lt:
        QA, absmax = quantize_jax(array)
        return cls(
            A=QA,
            absmax=absmax,
            shape=QA.shape,
            dtype=dtype or array.dtype,
        )

    @staticmethod
    def dequantize(
        A,
        absmax,
        float_dtype: jnp.dtype,
    ) -> Array:
        return dequantize_jax(
            A=A,
            absmax=absmax,
        ).astype(float_dtype)

    def __repr__(self) -> str:
        return f"Array4Lt(shape={self.shape}, dtype={self.dtype})"

    @property
    def nbytes(self) -> int:
        """
        Calculate the total number of bytes used by the quantized representation.

        Returns:
            int: The number of bytes used.
        """
        return self.array_quantized.nbytes + self.scale.nbytes + self.min_vals.nbytes

    def memory_savings(self) -> float:
        """
        Calculate the memory savings compared to the original array.

        Returns:
            float: The percentage of memory saved.
        """
        original_size = jnp.prod(jnp.array(self.shape)) * jnp.dtype(self.dtype).itemsize
        return (1 - self.nbytes / original_size) * 100


ArrayType = Union[Array, Array4Lt]


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

    Materializes Array4Lt inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        lhs (ArrayType): Left-hand side array.
        rhs (ArrayType): Right-hand side array.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.dot_general operation.
    """
    if isinstance(lhs, Array4Lt):
        lhs = lhs.materialize()
    if isinstance(rhs, Array4Lt):
        rhs = rhs.materialize()
    return lax.dot_general(lhs=lhs, rhs=rhs, *args, **kwargs)


@core.primitive_handler("add")
def handle_add(
    primitive,
    x: ArrayType,
    y: ArrayType,
):
    """
    Custom handler for JAX's add operation.

    Materializes Array4Lt inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        x (ArrayType): First array to add.
        y (ArrayType): Second array to add.

    Returns:
        The result of lax.add operation.
    """
    if isinstance(x, Array4Lt):
        x = x.materialize()
    if isinstance(y, Array4Lt):
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

    Materializes Array4Lt inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        operand (ArrayType): The array to be reduced.
        init_value (ArrayType): The initial value for the reduction.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.reduce operation.
    """
    if isinstance(operand, Array4Lt):
        operand = operand.materialize()
    if isinstance(init_value, Array4Lt):
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

    Materializes Array4Lt inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        x (ArrayType): First array to multiply.
        y (ArrayType): Second array to multiply.

    Returns:
        The result of lax.mul operation.
    """
    if isinstance(x, Array4Lt):
        x = x.materialize()
    if isinstance(y, Array4Lt):
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

    Materializes Array4Lt input before performing the operation.
    Re-quantizes the result if the input was Array4Lt.

    Args:
        primitive: The JAX primitive being handled.
        operand (ArrayType): The array to be transposed.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.transpose operation, potentially re-quantized.
    """
    original_quantized = False
    if isinstance(operand, Array4Lt):
        # # blocksize = operand.blocksize
        operand = operand.materialize()
        original_quantized = True
    operand = lax.transpose(operand, *args, **kwargs)
    if original_quantized:
        operand = Array4Lt.quantize(
            operand,
            # # blocksize=blocksize,
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

    Materializes Array4Lt inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        lhs (ArrayType): Left-hand side array (input).
        rhs (ArrayType): Right-hand side array (kernel).
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.conv operation.
    """
    if isinstance(lhs, Array4Lt):
        lhs = lhs.materialize()
    if isinstance(rhs, Array4Lt):
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

    Materializes Array4Lt inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        x (ArrayType): First array for max comparison.
        y (ArrayType): Second array for max comparison.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.max operation.
    """
    if isinstance(x, Array4Lt):
        x = x.materialize()
    if isinstance(y, Array4Lt):
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

    Materializes Array4Lt input before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        x (ArrayType): The array to apply exponential to.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.exp operation.
    """
    if isinstance(x, Array4Lt):
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

    Materializes Array4Lt input before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        x (ArrayType): The array to apply logarithm to.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.log operation.
    """
    if isinstance(x, Array4Lt):
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

    This function handles reshaping for both regular arrays and Array4Lt quantized arrays.
    It materializes Array4Bit input before reshaping and re-quantizes the result if the input was Array4Bit.

    Args:
        primitive (Primitive): The JAX primitive being handled.
        operand (ArrayType): The array to be reshaped.
        new_sizes (Tuple[int, ...]): The desired new shape of the array.
        dimensions (Tuple[int, ...], optional): The order in which dimensions should be permuted before reshaping.
        **kwargs: Additional keyword arguments for the reshape operation.

    Returns:
        ArrayType: The reshaped array, potentially re-quantized if the input was Array4Lt.

    Raises:
        ValueError: If the new shape is not compatible with the original array's size.
    """
    original_quantized = isinstance(operand, Array4Lt)

    if original_quantized:
        # blocksize = operand.blocksize
        operand = operand.materialize()

    try:
        reshaped = lax.reshape(operand, **kwargs)
    except ValueError as e:
        raise ValueError(
            f"Reshape operation failed: {str(e)}. "
            f"Ensure the new shape {kwargs} is compatible with the original array size."
        ) from e
    if original_quantized:
        reshaped = Array4Lt.quantize(
            operand,
            # blocksize=blocksize,
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

    Materializes Array4Lt inputs before performing the operation.

    Args:
        primitive: The JAX primitive being handled.
        operands (Sequence[ArrayType]): Sequence of arrays to concatenate.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of lax.concatenate operation.
    """
    materialized_operands = [
        op.materialize() if isinstance(op, Array4Lt) else op for op in operands
    ]
    return lax.concatenate(materialized_operands, *args, **kwargs)


@core.primitive_handler("convert_element_type")
def convert_element_type(
    primitive,
    arg: ArrayType,
    **params,
) -> ArrayType:
    """Handle element type conversion for Array4Lt."""
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
    """Handle broadcast_in_dim for Array4Lt."""
    original_quantized = isinstance(operand, Array4Lt)
    array = operand
    if original_quantized:
        # blocksize = operand.blocksize
        array = operand.materialize()
    result = jax.lax.broadcast_in_dim(array, *args, **kwargs)
    if original_quantized:
        result = Array4Lt.quantize(
            result,
            # blocksize=blocksize,
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
    """Handle gather for Array4Lt."""
    original_quantized = isinstance(operand, Array4Lt)
    array = operand
    if original_quantized:
        array = operand.materialize()
    result = jax.lax.gather(array, *args, **kwargs)
    return result
