from jax import numpy as jnp
import chex

MAX_INT8 = 127.5


def quantize_kv(kv: chex.Array):
    """Quantize key/values stored in kvcache."""
    scale = jnp.max(jnp.abs(kv), axis=-1, keepdims=True)
    value = jnp.int8(jnp.rint(kv * (MAX_INT8 / scale)))
    return value, scale


def unquantize_kv(value: chex.Array, scale: chex.Array, dtype: jnp.dtype):
    """Unquantize key/values stored in kvcache."""
    return value.astype(dtype) * scale / MAX_INT8
