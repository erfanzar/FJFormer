from functools import wraps
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from jax import tree_util
from jax.dtypes import float0

from fjformer.core.implicit_array import implicit_compact
from fjformer.core.symbols import SymbolicConstant


# Custom exceptions
class OptimizerError(Exception):
    """Base exception class for optimizer-related errors."""

    pass


class AxisError(OptimizerError):
    """Exception raised for errors related to axis operations."""

    pass


class FreezingError(OptimizerError):
    """Exception raised for errors during parameter freezing."""

    pass


class UpdateError(OptimizerError):
    """Exception raised for errors during parameter updates."""

    pass


def vmap_all_but_one(f: Callable, axis: int, out_ndim: int = 0) -> Callable:
    """
    Repeatedly calls vmap to map over all axes except for `axis`.
    All args will be mapped on the same dimensions.

    Args:
        f: The function to be mapped.
        axis: The axis to exclude from mapping.
        out_ndim: The number of dimensions in the output.

    Returns:
        A wrapped function that applies vmap to all axes except the specified one.

    Raises:
        AxisError: If the specified axis is out of bounds.
    """

    @wraps(f)
    def inner(*args: Any) -> Any:
        n_dim = args[0].ndim
        if axis >= n_dim:
            raise AxisError(
                f"Axis {axis} is out of bounds for array of dimension {n_dim}"
            )
        fn = f
        vmap_dim = 1
        out_dim = out_ndim
        for i in reversed(range(n_dim)):
            if i == axis:
                vmap_dim = 0
                out_dim = 0
            else:
                fn = jax.vmap(fn, vmap_dim, out_dim)
        return fn(*args)

    return inner


def freeze_subtrees(
    optimizer: optax.GradientTransformation,
    label_fn: Callable[[Any], Dict[str, Any]],
    use_scalar_zeros: bool = False,
) -> optax.GradientTransformation:
    """
    Wraps an optimizer such that subtrees specified by label_fn will receive zeros as updates.

    Args:
        optimizer: The original optimizer to be wrapped.
        label_fn: A function that labels subtrees as "freeze" or "train".
        use_scalar_zeros: If True, use scalar zeros instead of array zeros.

    Returns:
        A wrapped optimizer that freezes specified subtrees.

    Raises:
        FreezingError: If there's an error during the freezing process.
    """
    try:
        multi_transformed_optimizer = optax.multi_transform(
            {
                "freeze": (
                    set_to_zero_scalar() if use_scalar_zeros else optax.set_to_zero()
                ),
                "train": optimizer,
            },
            label_fn,
        )

        def new_update(grads: Any, opt_state: Any, params: Any) -> Tuple[Any, Any]:
            def map_float0(param: Any, grad: Any) -> Any:
                if grad.dtype == float0:
                    return (
                        jnp.zeros((), param.dtype)
                        if use_scalar_zeros
                        else jnp.zeros_like(param)
                    )
                return grad

            fixed_grads = jax.tree_util.tree_map(map_float0, params, grads)
            return multi_transformed_optimizer.update(fixed_grads, opt_state, params)

        return optax.GradientTransformation(
            multi_transformed_optimizer.init, new_update
        )
    except Exception as e:
        raise FreezingError(f"Error in freeze_subtrees: {str(e)}")


def freeze_keys(
    optimizer: optax.GradientTransformation,
    arr_type: type,
    keys: Union[List[str], Set[str]],
    use_scalar_zeros: bool = False,
) -> optax.GradientTransformation:
    """
    Freezes specific keys in the optimizer.

    Args:
        optimizer: The original optimizer to be wrapped.
        arr_type: The type of array to be processed.
        keys: A list or set of keys to be frozen.
        use_scalar_zeros: If True, use scalar zeros instead of array zeros.

    Returns:
        A wrapped optimizer with specified keys frozen.

    Raises:
        FreezingError: If there's an error during the freezing process.
    """
    try:
        keys_set = set(keys)

        def label_leaf(leaf: Any) -> Union[str, Any]:
            if not isinstance(leaf, arr_type):
                return "train"

            children, aux_data = leaf.tree_flatten_with_keys()
            labels = ["freeze" if key in keys_set else "train" for key, _ in children]
            struct = leaf.tree_unflatten(aux_data, labels)
            return struct

        def label_fn(root: Any) -> Any:
            return jax.tree_util.tree_map(
                label_leaf, root, is_leaf=lambda x: isinstance(x, arr_type)
            )

        return freeze_subtrees(optimizer, label_fn, use_scalar_zeros=use_scalar_zeros)
    except Exception as e:
        raise FreezingError(f"Error in freeze_keys: {str(e)}")


def apply_updates(params: optax.Params, updates: optax.Updates) -> optax.Params:
    """
    Applies updates to parameters, supporting SymbolicConstant instances.

    Args:
        params: The current parameters.
        updates: The updates to be applied.

    Returns:
        The updated parameters.

    Raises:
        UpdateError: If there's an error during the update process.
    """
    try:
        updates_flat, update_struct = tree_util.tree_flatten(
            updates, is_leaf=lambda x: isinstance(x, SymbolicConstant)
        )
        semi_flat_params = update_struct.flatten_up_to(params)

        updated_flat = implicit_compact(optax.apply_updates)(
            semi_flat_params, updates_flat
        )
        updated = update_struct.unflatten(updated_flat)
        return updated
    except Exception as e:
        raise UpdateError(f"Error in apply_updates: {str(e)}")


def set_to_zero_scalar() -> optax.GradientTransformation:
    """
    Returns a gradient transformation that sets all gradients to 0 to make downstream constant folding cheaper.

    Returns:
        A gradient transformation that sets all gradients to scalar zeros.

    Raises:
        OptimizerError: If there's an error during the transformation process.
    """
    try:

        def init_fn(params: Any) -> optax.EmptyState:
            del params
            return optax.EmptyState()

        def update_fn(updates: Any, state: Any, params: Any = None) -> Tuple[Any, Any]:
            return (
                jax.tree_util.tree_map(lambda x: jnp.zeros((), x.dtype), updates),
                state,
            )

        return optax.GradientTransformation(init_fn, update_fn)
    except Exception as e:
        raise OptimizerError(f"Error in set_to_zero_scalar: {str(e)}")
