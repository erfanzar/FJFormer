import chex
import jax
from jax import numpy as jnp


def global_norm(tree):
    """ Return the global L2 norm of a pytree. """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = jax.flatten_util.ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))


def average_metrics(metrics):
    """
    The average_metrics function takes a list of metrics and averages them.

    :param metrics: Store the metrics for each batch
    :return: The mean of the metrics across all runs
    
    """
    return jax.tree_map(
        lambda *args: jnp.mean(jnp.stack(args)),
        *metrics
    )


def transpose(array: chex.Array, dim0: int, dim1: int):
    """
    The transpose function takes an array and two dimensions, and returns a new
    array with the specified dimensions transposed. The first dimension is given as
    a positive integer, where 0 represents the outermost dimension of the array. If
    the first dimension is negative, it counts from the end of the shape tuple; -2
    is equivalent to len(shape) - 2. The second dimension may be specified in a similar way.

    :param array: chex.Array: Specify the array to be transposed
    :param dim0: int: Specify the first dimension to be transposed
    :param dim1: int: Specify the dimension of the array
    :return: A new array with the same data, but with axes permuted
    
    """
    dim0 = dim0 if dim0 > 0 else array.ndim - dim0
    dim1 = dim1 if dim1 > 0 else array.ndim - dim1
    perm = list(range(array.ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return jnp.transpose(array, perm)


def fused_softmax(x: chex.Array, axis: int = -1):
    """
    The fused_softmax function is a fused version of the softmax function.

    :param x: chex.Array: Specify the input to the function
    :param axis: int: Specify the axis along which to apply the softmax function
    :return: The same result as the softmax function
    
    """
    return jnp.exp(jax.nn.log_softmax(x, axis=axis))
