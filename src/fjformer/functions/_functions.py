import chex
import jax
from jax import numpy as jnp
from jax._src.flatten_util import ravel_pytree


def global_norm(tree: chex.ArrayTree) -> chex.Array:
    """
    Computes the global norm of a PyTree of arrays.

    The global norm is the square root of the sum of squares of all elements in the PyTree.

    Args:
        tree: The PyTree of arrays.

    Returns:
        The global norm as a scalar JAX array.
    """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))


def average_metrics(metrics: list[chex.ArrayTree]) -> chex.ArrayTree:
    """
    Averages a list of metrics, each represented as a PyTree.

    This function is useful for averaging metrics across multiple devices or steps.

    Args:
        metrics: A list of PyTrees representing the metrics.

    Returns:
        A PyTree with the same structure as the input metrics, but with each leaf
        node replaced by the average of the corresponding leaf nodes across the
        input metrics.
    """
    return jax.tree_util.tree_map(lambda *args: jnp.mean(jnp.stack(args)), *metrics)


def transpose(array: chex.Array, dim0: int, dim1: int) -> chex.Array:
    """
    Transposes two dimensions of a JAX array.

    This function provides a convenient way to swap two dimensions of an array using
    negative indexing for dimensions.

    Args:
        array: The input array.
        dim0: The first dimension to transpose. Can be negative.
        dim1: The second dimension to transpose. Can be negative.

    Returns:
        The transposed array.
    """
    dim0 = dim0 if dim0 > 0 else array.ndim + dim0
    dim1 = dim1 if dim1 > 0 else array.ndim + dim1
    perm = list(range(array.ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return jnp.transpose(array, perm)


def fused_softmax(x: chex.Array, axis: int = -1) -> chex.Array:
    """
    Computes a numerically stable softmax using a fused implementation.

    This function computes the softmax by first calculating the log-softmax and
    then exponentiating the result. This approach is more numerically stable than
    directly computing the softmax.

    Args:
        x: The input array.
        axis: The axis along which to compute the softmax.

    Returns:
        The softmax of the input array.
    """
    return jnp.exp(jax.nn.log_softmax(x, axis=axis))
