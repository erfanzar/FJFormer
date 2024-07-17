from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp

from fjformer.core.errors import (
    MaterializationError,
    OperationError,
    ShapeDtypeError,
    UnsupportedPrimitiveError,
)
from fjformer.core.implicit_array import (
    ELEMENTWISE_BINOPS,
    ELEMENTWISE_UNOPS,
    ArrayValue,
    ImplicitArray,
    aux_field,
    default_handler,
    implicit_compact,
    primitive_handler,
)
from fjformer.core.types import Complement

INFINITY = float("inf")
NEGATIVE_INFINITY = float("-inf")
_GENERAL = -2
_SPECIALIZED = -1


def _get_shape_dtype(
    x: Any, shape: Optional[Tuple[int, ...]] = None, dtype: Optional[Any] = None
) -> Tuple[Tuple[int, ...], Any]:
    """
    Determine the shape and dtype for a given input.

    Args:
        x: Input array or value
        shape: Optional shape to use
        dtype: Optional dtype to use

    Returns:
        A tuple containing:
            - shape: Tuple[int, ...]: The determined shape
            - dtype: Any: The determined dtype

    Raises:
        ShapeDtypeError: If the input x is None and both shape and dtype are None
    """
    if x is None and shape is None and dtype is None:
        raise ShapeDtypeError("Either x or both shape and dtype must be provided")

    try:
        if shape is None:
            shape = np.shape(x)
        else:
            shape = jax.core.canonicalize_shape(shape)

        if dtype is None:
            dtype = lax.dtype(x)

        return shape, dtype
    except Exception as e:
        raise ShapeDtypeError(f"Error determining shape and dtype: {str(e)}")


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
        partial(default_handler, primitive, **kwargs),
        *(jax.core.get_aval(x) for x in args),
    )
    return jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), out_aval)


def symbolic_zero_like(
    x: Any, shape: Optional[Tuple[int, ...]] = None, dtype: Optional[Any] = None
) -> "SymbolicConstant":
    """
    Create a SymbolicConstant filled with zeros, similar to the input.

    Args:
        x: Input to base the result on
        shape: Optional shape for the result
        dtype: Optional dtype for the result

    Returns:
        SymbolicConstant: A constant array filled with zeros

    Raises:
        OperationError: If creation of the zero-filled SymbolicConstant fails
    """
    try:
        dtype = lax.dtype(x) if dtype is None else dtype
        return symbolic_full_like(x, 0, shape=shape, dtype=dtype)
    except Exception as e:
        raise OperationError(f"Failed to create zero-filled SymbolicConstant: {str(e)}")


def symbolic_full_like(
    x: Any,
    fill_value: Any,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[Any] = None,
) -> "SymbolicConstant":
    """
    Create a SymbolicConstant filled with a specific value, similar to the input.

    Args:
        x: Input to base the result on
        fill_value: Value to fill the SymbolicConstant with
        shape: Optional shape for the result
        dtype: Optional dtype for the result

    Returns:
        SymbolicConstant filled with the specified value
    """
    shape, _ = _get_shape_dtype(x, shape, None)
    if dtype is None:
        dtype = jax.lax.dtype(fill_value)

    return SymbolicConstant(fill_value, shape=shape, dtype=dtype)


@dataclass
class SymbolicConstant(ImplicitArray):
    """
    Represents a constant array symbolically, allowing for efficient operations at compile time.

    Attributes:
        value (Any): The constant value of the array
        weak_type (bool): Whether the constant has a weak type
    """

    value: Any = aux_field()
    weak_type: bool = aux_field(default=False)

    def __post_init__(self):
        """Initialize the SymbolicConstant after creation."""
        super().__post_init__()
        try:
            with jax.ensure_compile_time_eval():
                self.value = jnp.asarray(self.value, dtype=self.dtype)
        except Exception as e:
            raise OperationError(f"Failed to initialize SymbolicConstant: {str(e)}")

    def compute_dtype(self) -> Any:
        """Compute the dtype of the constant."""
        return lax.dtype(self.value)

    def materialize(self) -> jnp.ndarray:
        """
        Materialize the symbolic constant into a concrete array.

        Returns:
            jnp.ndarray: A concrete JAX array

        Raises:
            MaterializationError: If materialization fails
        """
        try:
            return jnp.full(self.shape, self.value, dtype=self.dtype)
        except Exception as e:
            raise MaterializationError(
                f"Failed to materialize SymbolicConstant: {str(e)}"
            )

    def copy(self) -> "SymbolicConstant":
        """
        Create a copy of the SymbolicConstant.

        Returns:
            SymbolicConstant: A copy of the current instance

        Raises:
            OperationError: If copying fails
        """
        try:
            return jax.tree_util.tree_map(lambda x: x, self)
        except Exception as e:
            raise OperationError(f"Failed to copy SymbolicConstant: {str(e)}")


@implicit_compact
def broadcast_to(val, shape):
    return jnp.broadcast_to(val, shape)


@implicit_compact
def astype(val, dtype):
    return val.astype(dtype)


@primitive_handler(
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
    sym: SymbolicConstant,
    **kwargs: Any,
) -> SymbolicConstant:
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
    return SymbolicConstant(sym.value, shape=out_shape, dtype=out_dtype)


def _op_and_reshape(primitive, lhs, rhs, flip=False):
    """
    Close over one arg so we can do math at tracing time, but let the other one get traced.
    """
    if flip:
        lhs, rhs = (rhs, lhs)

    @implicit_compact
    def inner(arg):
        other = lhs
        if flip:
            arg, other = (other, arg)

        result = default_handler(primitive, arg, other)
        return result

    return inner(rhs)


def special_case_binop(
    name,
    identity=None,
    annihilator=None,
    flip=False,
):
    """
    Create a special case handler for binary operations.
    """
    lhs_type = SymbolicConstant
    rhs_type = Complement[ArrayValue, SymbolicConstant]
    if flip:
        lhs_type, rhs_type = rhs_type, lhs_type

    @primitive_handler(name, precedence=_SPECIALIZED)
    def handler(primitive, lhs: lhs_type, rhs: rhs_type, **kwargs):  # type:ignore
        out_shape, out_dtype = _out_shape_dtype(
            primitive,
            lhs,
            rhs,
            **kwargs,
        )
        with jax.ensure_compile_time_eval():
            if lhs.value == identity:
                return broadcast_to(astype(rhs, out_dtype), out_shape)

            if lhs.value == annihilator:
                return SymbolicConstant(lhs.value, shape=out_shape, dtype=out_dtype)

            print(f"{primitive} {lhs.value} {rhs}")
            return _op_and_reshape(primitive, lhs.value, rhs)


# Define special case handlers for various binary operations
special_case_binop("add", identity=0)
special_case_binop("mul", identity=1, annihilator=0)
special_case_binop("and", annihilator=0)
special_case_binop("or", identity=0)
special_case_binop("xor", identity=0)
special_case_binop("sub", identity=0, flip=True)
special_case_binop("div", identity=1, flip=True)
special_case_binop("exp", identity=1, flip=True)
special_case_binop("min", identity=float("inf"), annihilator=float("-inf"))
special_case_binop("max", identity=float("-inf"), annihilator=float("inf"))


def eval_default_handler(primitive, *args, **kwargs):
    """
    Evaluate the default handler for a primitive at compile time.
    """
    with jax.ensure_compile_time_eval():
        result = primitive.bind(*args, **kwargs)
    return result


@primitive_handler(ELEMENTWISE_UNOPS, precedence=_GENERAL)
def handle_unop(primitive, sym: SymbolicConstant, **kwargs):
    """
    Handle unary operations on SymbolicConstants.

    Args:
        primitive: The JAX primitive to handle
        sym: The SymbolicConstant to operate on
        **kwargs: Additional keyword arguments

    Returns:
        SymbolicConstant: The result of the unary operation

    Raises:
        UnsupportedPrimitiveError: If the primitive is not supported
        OperationError: If the operation fails
    """

    try:
        new_val = eval_default_handler(primitive, sym.value, **kwargs)
        return symbolic_full_like(sym, new_val)
    except Exception as e:
        if isinstance(e, NotImplementedError):
            raise UnsupportedPrimitiveError(f"Unsupported primitive: {primitive}")
        raise OperationError(f"Failed to handle unary operation {primitive}: {str(e)}")


@primitive_handler(ELEMENTWISE_BINOPS, precedence=_GENERAL)
def handle_binop(
    primitive,
    lhs: SymbolicConstant,
    rhs: SymbolicConstant,
    **kwargs,
):
    """
    Handle binary operations on SymbolicConstants.
    """
    out_shape, out_dtype = _out_shape_dtype(primitive, lhs, rhs, **kwargs)
    new_val = eval_default_handler(primitive, lhs.value, rhs.value, **kwargs)
    return symbolic_full_like(lhs, new_val, shape=out_shape, dtype=out_dtype)


@primitive_handler(["reduce_sum", "reduce_prod"])
def reduce_sum(
    primitive,
    sym: SymbolicConstant,
    *,
    axes,
):
    """
    Handle reduction operations (sum and product) on SymbolicConstants.
    """
    out_shape, out_dtype = _out_shape_dtype(primitive, sym, axes=axes)
    with jax.ensure_compile_time_eval():
        if sym.value == 0:
            return SymbolicConstant(0, shape=out_shape, dtype=out_dtype)

        orig_size = np.prod(sym.shape)
        new_size = np.prod(out_shape)

        n_combined = orig_size // new_size

        new_val = sym.value
        if primitive.name == "reduce_sum":
            new_val = new_val * n_combined
        else:
            new_val = new_val**n_combined

    return SymbolicConstant(new_val, shape=out_shape, dtype=out_dtype)


@primitive_handler("select_n")
def handle_select_n(
    primitive,
    cond_val,
    *arg_vals: SymbolicConstant,
):
    """
    Handle select_n operation on SymbolicConstants.
    """
    if len(set(val.value.item() for val in arg_vals)) != 1:
        return NotImplemented

    out_shape, out_dtype = _out_shape_dtype(primitive, cond_val, *arg_vals)
    return SymbolicConstant(arg_vals[0].value, shape=out_shape, dtype=out_dtype)
