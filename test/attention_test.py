import math
import os

import jax

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
from flax.linen.attention import dot_product_attention, make_attention_mask, make_causal_mask, combine_masks

from src.fjformer.pallas_operations import flash_attention
from jax import random, numpy as jnp

batch = 1
seq = 6
heads = 32
hd = 128


def combine_causal_and_attention_mask(q_attention_mask, kv_attention_mask):
    assert q_attention_mask.ndim == 2
    assert kv_attention_mask.ndim == 2
    return jnp.bitwise_and(
        jnp.bitwise_and(q_attention_mask.astype("bool"), kv_attention_mask.astype("bool")),
        jnp.tril(jnp.ones(
            (q_attention_mask.shape[-1], kv_attention_mask.shape[-1]), dtype="bool")
        )[None, None, :, :]
    ).astype("bool")


def main():
    k1, k2, k3 = random.split(random.PRNGKey(3), num=3)
    q = random.normal(k1, (batch, seq, heads, hd), "float32")
    k = random.normal(k2, (batch, seq, heads, hd), "float32")
    v = random.normal(k3, (batch, seq, heads, hd), "float32")
    a = jnp.ones((batch, seq), dtype="bool")
    a = a.at[:, :seq // 2].set(0)
    # a = a.at[:, seq // 2:].set(0)
    # print(a)
    csm = make_causal_mask(jnp.ones((batch, seq)))
    mask = combine_masks(csm, a[:, None, None, :])
    b = jnp.where(mask, 0, jnp.finfo(jnp.float32).min)
    out = dot_product_attention(q, k, v, b)
    cnk = seq // 2
    out_flash = flash_attention(
        q,
        k,
        v,
        b,
        # sm_scale=1 / math.sqrt(hd),
        block_k=cnk,
        block_q=cnk,
    )
    print(jnp.mean(jnp.sum(out_flash)))
    print(jnp.mean(jnp.sum(out)))
    # am = jnp.bitwise_and(c[None, None, :, :], a[:, None, None, :])
    # print(am.on_device_size_in_bytes())
    # msk = combine_causal_and_attention_mask(a, a)
    # df = pd.DataFrame(mask.astype("int").reshape(seq, seq))
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(df, annot=True, cmap='coolwarm', cbar=False, linewidths=0.0, fmt='d')
    # plt.title('Attention Mask')
    # plt.xlabel('Sequence Position')
    # plt.ylabel('Sequence Position')
    # plt.show()


if __name__ == '__main__':
    main()
