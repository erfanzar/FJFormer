# jax.experimental.pallas.ops.gpu

"""Pallas softmax kernel."""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _vmappable_softmax_kernel(
        # inputs
        input_ref,
        # outputs
        probs_ref,
        *,
        # block information
        # It is assumed that block_row >= row_len
        block_row: int,
):
    row_len = input_ref.shape[-1]

    mask = jnp.arange(block_row) < row_len
    row = pl.load(
        input_ref, (pl.dslice(0, block_row),), mask=mask, other=-float("inf")
    )

    row_max = jnp.max(row, axis=0)
    numerator = jnp.exp((row - row_max).astype(jnp.float32))
    denominator = jnp.sum(numerator, axis=0)

    pl.store(
        probs_ref, (pl.dslice(0, block_row),),
        (numerator / denominator).astype(probs_ref.dtype),
        mask=mask
    )


@functools.partial(jax.jit, static_argnames=["axis", "num_warps", "interpret",
                                             "debug"])
def softmax(
        x: jax.Array, *, axis: int = -1, num_warps: int = 4,
        interpret: bool = False, debug: bool = False
) -> jax.Array:
    """Computes the softmax of the input array along the specified axis.

    Args:
      x: input array
      axis: the axis along which to perform the computation
      num_warps: the number of warps to use for executing the Triton kernel
      interpret: whether to interpret the kernel using pallas
      debug: whether to use pallas in debug mode

    Returns:
      The result of the softmax operation over the specified axis of x.
    """
    axis = axis if axis >= 0 else len(x.shape) + axis
    if axis != len(x.shape) - 1:
        raise NotImplementedError(
            "reductions along non-trailing dimension unsupported")

    row_len = x.shape[-1]

    block_row = pl.next_power_of_2(row_len)
    out_shape = jax.ShapeDtypeStruct(shape=(row_len,), dtype=x.dtype)

    kernel = functools.partial(_vmappable_softmax_kernel, block_row=block_row)
    f = pl.pallas_call(
        kernel,
        compiler_params=dict(triton=dict(num_warps=num_warps, num_stages=1)),
        grid=(),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
    )

    for _ in range(len(x.shape) - 1):
        f = jax.vmap(f)

    return f(x)
