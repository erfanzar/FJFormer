import chex
import jax
from jax import numpy as jnp


def global_norm(tree):
    """ Return the global L2 norm of a pytree. """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = jax.flatten_util.ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))


def average_metrics(metrics):
    return jax.tree_map(
        lambda *args: jnp.mean(jnp.stack(args)),
        *metrics
    )


def transpose(array: chex.Array, dim0: int, dim1: int):
    dim0 = dim0 if dim0 > 0 else array.ndim - dim0
    dim1 = dim1 if dim1 > 0 else array.ndim - dim1
    perm = list(range(array.ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return jnp.transpose(array, perm)


def fused_softmax(x: chex.Array, axis: int = -1):
    return jnp.exp(jax.nn.log_softmax(x, axis=axis))
