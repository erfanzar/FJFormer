# TODO : Implement Custom Backward Prp
import functools
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import jax
from jax import Array, lax
from jax import numpy as jnp
from jax.core import Primitive

import fjformer.core as core


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
        min_vals (core.ArrayValue): Minimum values used in quantization.
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
        ...     return jnp.dot(a, b)

        >>> result = f(x, xp)
        >>> q_result = f(quantized_x, quantized_xp)

        >>> print(jnp.allclose(result, q_result, rtol=1e-2, atol=1e-2))
        True
    """

    array_quantized: core.ArrayValue
    scale: core.ArrayValue
    min_vals: core.ArrayValue

    def materialize(self) -> Array:
        """
        Materialize the quantized array back to its original representation.

        Returns:
            Array: The dequantized array in its original dtype.
        """
        return self.dequantize(
            array_quantized=self.array_quantized,
            scale=self.scale,
            min_vals=self.min_vals,
            float_dtype=self.dtype,
        )

    @classmethod
    def quantize(
        cls,
        array: Array,
        axis: int = -1,
        dtype: Optional[jnp.dtype] = None,
    ) -> "Array8Bit":
        """
        Quantize a JAX array to 8-bit representation.

        Args:
            array (Array): The input array to quantize.
            axis (int, optional): The axis along which to compute min and max. Defaults to -1.
            dtype (jnp.dtype, optional): The desired dtype for the output. If None, uses the input array's dtype.

        Returns:
            Array8Bit: The quantized array.
        """
        min_vals = jnp.min(array, axis=axis, keepdims=True)
        max_vals = jnp.max(array, axis=axis, keepdims=True)

        # Compute the scaling factors
        scale = (max_vals - min_vals) / 255

        # Quantize the data
        quantized_data = jnp.round((array - min_vals) / scale)

        # Clip the quantized values to ensure they lie within the representable range
        quantized_data = jnp.clip(quantized_data, 0, 255).astype(jnp.uint8)

        return cls(
            array_quantized=quantized_data,
            scale=scale,
            min_vals=min_vals,
            shape=quantized_data.shape,
            dtype=dtype or array.dtype,
        )

    @staticmethod
    def dequantize(
        array_quantized: Array,
        scale: Array,
        min_vals: Array,
        float_dtype: jnp.dtype,
    ) -> Array:
        """
        Dequantize an 8-bit array back to its original representation.

        Args:
            array_quantized (Array): The quantized array data.
            scale (Array): The scaling factors used in quantization.
            min_vals (Array): The minimum values used in quantization.
            float_dtype (jnp.dtype): The desired output dtype.

        Returns:
            Array: The dequantized array.
        """
        return (array_quantized.astype(float_dtype) * scale + min_vals).astype(
            float_dtype
        )

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
        min_vals_slice = self.min_vals[idx] if self.min_vals.ndim > 0 else self.min_vals
        return self.dequantize(quantized_slice, scale_slice, min_vals_slice, self.dtype)

    def __repr__(self) -> str:
        return f"Array8Bit(shape={self.shape}, dtype={self.dtype})"

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
    return lax.dot_general(lhs=lhs, rhs=rhs, *args, **kwargs)


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
    It materializes Array4Bit input before reshaping and re-quantizes the result if the input was Array4Bit.

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
