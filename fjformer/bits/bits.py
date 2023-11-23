import builtins

import jax
from jax import core
from jax import lax
import jax.numpy as jnp
import numpy as np

shape = np.shape
ndim = np.ndim
_max = builtins.max


def matmul(
        a: jnp.ndarray,
        b: jnp.ndarray,
        *,
        precision=None,
        dot_general=lax.dot_general
) -> jnp.ndarray:
    """Quantized jax.numpy.matmul.

    Args:
      a: Left-hand side of the matmul.
      b: Right-hand side of the matmul.
      precision: Indicates precision of a and b.
      dot_general: lax.dot_general by default. To use quantized matmul, the
        wrapper of q_dot_general in which TQs and `train` flag are provided
        should be passed into this function.

    Returns:
      An array containing the result with the same dtype as 'a' and 'b'.
    """
    arraylike = (jax.Array, np.ndarray)
    if not isinstance(a, arraylike) or not isinstance(b, arraylike):
        raise TypeError(f"matmul requires array-like arguments, got {a} and {b}")
    for i, x in enumerate((a, b)):
        if ndim(x) < 1:
            msg = (f"matmul input operand {i} must have ndim at least 1, "
                   f"but it has ndim {ndim(x)}")
            raise ValueError(msg)

    dtype = jnp.result_type(a.dtype, b.dtype)
    a = a.astype(dtype)
    b = b.astype(dtype)

    a_is_mat, b_is_mat = (ndim(a) > 1), (ndim(b) > 1)
    a_batch_dims = shape(a)[:-2] if a_is_mat else ()
    b_batch_dims = shape(b)[:-2] if b_is_mat else ()
    num_batch_dims = _max(len(a_batch_dims), len(b_batch_dims))
    a_batch_dims = (None,) * (num_batch_dims - len(a_batch_dims)) + a_batch_dims
    b_batch_dims = (None,) * (num_batch_dims - len(b_batch_dims)) + b_batch_dims

    # Dimensions to squeeze from the inputs.
    a_squeeze = []
    b_squeeze = []

    # Positions of batch dimensions in squeezed inputs.
    a_batch = []
    b_batch = []

    # Desired index in final output of each kind of dimension, in the order that
    # aqt_dot_general will emit them.
    idx_batch = []
    idx_a_other = []  # other = non-batch, non-contracting.
    idx_b_other = []
    for i, (ba, bb) in enumerate(zip(a_batch_dims, b_batch_dims)):
        if ba is None:
            idx_b_other.append(i)
        elif bb is None:
            idx_a_other.append(i)
        elif core.symbolic_equal_dim(ba, 1):
            idx_b_other.append(i)
            a_squeeze.append(len(idx_batch) + len(idx_a_other) + len(a_squeeze))
        elif core.symbolic_equal_dim(bb, 1):
            idx_a_other.append(i)
            b_squeeze.append(len(idx_batch) + len(idx_b_other) + len(b_squeeze))
        elif core.symbolic_equal_dim(ba, bb):
            a_batch.append(len(idx_batch) + len(idx_a_other))
            b_batch.append(len(idx_batch) + len(idx_b_other))
            idx_batch.append(i)
        else:
            raise ValueError("Incompatible shapes for matmul arguments: {} and {}"
                             .format(shape(a), shape(b)))

    if a_is_mat:
        idx_a_other.append(num_batch_dims)
    if b_is_mat:
        idx_b_other.append(num_batch_dims + a_is_mat)
    perm = np.argsort(np.concatenate([idx_batch, idx_a_other, idx_b_other]))

    a = lax.squeeze(a, tuple(a_squeeze))
    b = lax.squeeze(b, tuple(b_squeeze))
    out = dot_general(
        a,
        b, (((ndim(a) - 1,), (ndim(b) - 1 - b_is_mat,)), (a_batch, b_batch)),
        precision=precision)
    return lax.transpose(out, perm)


def matmul_true_int8(lhs, rhs):
    assert lhs.dtype == jnp.int8
    assert rhs.dtype == jnp.int8
    result = jnp.matmul(lhs, rhs, preferred_element_type=jnp.int32)
    assert result.dtype == jnp.int32
    return result


def quant_int8(x):
    return jnp.clip(jnp.round(x), -127, 127).astype(jnp.int8)


def q_matmul_int8(a, w):

    # Calibration. Calibration function is also customizable and injectable.
    a_s = 127 / jnp.max(jnp.abs(a), axis=1, keepdims=True)
    w_s = 127 / jnp.max(jnp.abs(w), axis=0, keepdims=True)

    # int8 matmul with int32 accumulator
    result = matmul_true_int8(quant_int8(a * a_s), quant_int8(w * w_s)) / (a_s * w_s)

    return result
