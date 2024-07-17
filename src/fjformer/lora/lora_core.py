import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp

from fjformer import core as cr


# Custom exceptions
class LoraError(Exception):
    """Base exception for LoRA-related errors."""

    pass


class UnsupportedOperationError(LoraError):
    """Raised when an unsupported operation is encountered."""

    pass


def lora(f: Any) -> Any:
    """Decorator for LoRA-compatible functions."""
    return cr.implicit_compact(f)


@dataclass
class LoraWeight(cr.ImplicitArray):
    """Represents a LoRA (Low-Rank Adaptation) weight."""

    w: cr.ArrayValue  # M x N (2D)
    a: cr.ArrayValue  # k x N (2D)
    b: cr.ArrayValue  # M x k (2D)
    alpha: float = cr.aux_field(default=1.00)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (
            self.a.shape[-2] == self.b.shape[-1]
            and self.w.shape[-2] == self.b.shape[-2]
            and self.w.shape[-1] == self.a.shape[-1]
        ):
            raise LoraError("Incompatible shapes in LoraWeight initialization")

    def materialize(self) -> jnp.ndarray:
        """Materialize the LoRA weight."""
        return (self.w + self.get_scale() * self.b @ self.a).astype(self.w.dtype)

    def get_scale(self) -> float:
        """Get the scaling factor for LoRA."""
        return self.alpha / self.b.shape[-1]


def _check_dot_dimension_numbers(dimension_numbers: Any) -> bool:
    """Check if the dimension numbers are supported for dot product."""
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    if lhs_batch or rhs_batch:
        warnings.warn("Batched matmuls are not supported")
        return False
    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        warnings.warn("Only matrix multiplication is supported")
        return False
    return True


@cr.primitive_handler("dot_general")
def handle_dot_lhs(
    primitive: Any,
    lora: LoraWeight,
    rhs: cr.ArrayValue,
    *,
    dimension_numbers: Any,
    **kwargs: Any,
) -> Union[jnp.ndarray, Any]:
    """Handle dot product when LoraWeight is on the left-hand side."""
    if not _check_dot_dimension_numbers(dimension_numbers):
        return NotImplemented

    if isinstance(rhs, LoraWeight):
        rhs = rhs.materialize()
        warnings.warn("Encountered product of two LoraWeights. Materializing the rhs")

    op = partial(jax.lax.dot_general, **kwargs)
    (lhs_contract,) = dimension_numbers[0][0]
    first, second = (lora.a, lora.b) if lhs_contract == 1 else (lora.b, lora.a)
    first *= lora.get_scale()

    orig = op(lora.w, rhs, dimension_numbers=dimension_numbers)
    lora_product = op(first, rhs, dimension_numbers=dimension_numbers)
    second_dimension_numbers = ((lhs_contract,), (0,)), dimension_numbers[1]
    lora_product = op(second, lora_product, dimension_numbers=second_dimension_numbers)

    return (orig + lora_product).astype(orig.dtype)


@cr.primitive_handler("dot_general")
def handle_dot_rhs(
    primitive: Any,
    lhs: jax.Array,
    lora: LoraWeight,
    *,
    dimension_numbers: Any,
    **kwargs: Any,
) -> Union[jnp.ndarray, Any]:
    """Handle dot product when LoraWeight is on the right-hand side."""
    if not _check_dot_dimension_numbers(dimension_numbers):
        return NotImplemented
    op = partial(jax.lax.dot_general, **kwargs)

    (rhs_contract,) = dimension_numbers[0][1]
    first, second = (lora.a, lora.b) if rhs_contract == 1 else (lora.b, lora.a)
    first *= lora.get_scale()

    orig = op(lhs, lora.w, dimension_numbers=dimension_numbers)
    lora_product = op(lhs, first, dimension_numbers=dimension_numbers)
    second_dimension_numbers = ((lhs.ndim - 1), (rhs_contract,)), dimension_numbers[1]
    lora_product = op(lora_product, second, dimension_numbers=second_dimension_numbers)

    return (orig + lora_product).astype(orig.dtype)


@cr.primitive_handler("conv_general_dilated")
def handle_conv(
    primitive: Any,
    inp: cr.ArrayValue,
    lora: LoraWeight,
    *,
    dimension_numbers: Any,
    **params: Any,
) -> jnp.ndarray:
    """Handle convolution with LoraWeight."""
    if isinstance(inp, LoraWeight):
        warnings.warn(
            "Using a LoraWeight as input to a convolution is not supported, so it will be materialized."
        )
        inp = inp.materialize()

    if dimension_numbers.rhs_spec[:1] != (
        len(dimension_numbers.rhs_spec) - 1,
        len(dimension_numbers.rhs_spec) - 2,
    ):
        raise UnsupportedOperationError(
            "LoraWeight only supports convolutions with shape (..., in_features, out_features)"
        )

    params = {**params, "dimension_numbers": dimension_numbers}
    op = partial(jax.lax.conv_general_dilated, **params)
    orig = op(inp, lora.w)
    lora_product = op(inp, lora.b)

    params["window_strides"] = (1,) * (len(dimension_numbers.rhs_spec) - 2)
    params["padding"] = "VALID"
    lora_product = jax.lax.conv_general_dilated(
        lora_product, lora.a * lora.get_scale(), **params
    )

    return (orig + lora_product).astype(orig.dtype)


@cr.primitive_handler("gather")
def handle_gather(
    primitive: Any,
    lora: LoraWeight,
    indices: jax.Array,
    *,
    dimension_numbers: Any,
    slice_sizes: Tuple[int, ...],
    **params: Any,
) -> Union[jnp.ndarray, Any]:
    """Handle gather operation with LoraWeight."""
    if dimension_numbers.offset_dims != (len(indices.shape) - 1,):
        return NotImplemented

    lora_dim = lora.b.shape[-1]

    if slice_sizes != (1, lora.a.shape[1]):
        return NotImplemented

    params = {**params, "dimension_numbers": dimension_numbers}
    orig = jax.lax.gather(lora.w, indices, slice_sizes=slice_sizes, **params)
    new_slice_sizes = (1, lora_dim)
    lora_product = jax.lax.gather(
        lora.b, indices, slice_sizes=new_slice_sizes, **params
    )
    lora_product = lora_product @ (lora.a * lora.get_scale())

    return (orig + lora_product).astype(orig.dtype)


@cr.primitive_handler("transpose")
def eval_lora_transpose(
    primitive: Any, arg: LoraWeight, *, permutation: Tuple[int, ...]
) -> Union[LoraWeight, Any]:
    """Handle transpose operation for LoraWeight."""
    if not (len(arg.shape) == 2 and permutation == (1, 0)):
        return NotImplemented

    return LoraWeight(
        w=arg.w.T,
        a=arg.b.T,
        b=arg.a.T,
        alpha=arg.alpha,
    )


@cr.primitive_handler("convert_element_type")
def eval_lora_convert_element_type(
    primitive: Any, arg: LoraWeight, **params: Any
) -> LoraWeight:
    """Handle element type conversion for LoraWeight."""
    result = jax.tree_util.tree_map(
        partial(cr.default_handler, primitive, **params), arg
    )
    result.dtype = params["new_dtype"]
    return result
