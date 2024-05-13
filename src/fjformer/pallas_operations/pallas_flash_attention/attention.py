# Modified Implementation of Flash attention or MHA from org jax authors
from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import functools
import math
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def flash_attention_forward_kernel(
        q_ref,
        k_ref,
        v_ref,  # Input arrays
        attention_mask_ref: jax.Array | None,  # segment_id arrays
        o_ref: Any,  # Output
        *residual_refs: Any,  # Residual outputs
        num_heads: int,
        sm_scale: float,
        causal: bool,
        block_q: int,
        block_d: int,
        block_k: int,
):
    seq_len = q_ref.shape[0]
    start_q = pl.program_id(0)

    m_i = jnp.zeros(block_q, dtype=jnp.float32) - float("inf")
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    o = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = pl.load(q_ref, (curr_q_slice, pl.dslice(None)))
    q_attention_mask = (
        None
        if attention_mask_ref is None
        else pl.load(attention_mask_ref, (curr_q_slice,))
    )

    def body(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pl.load(k_ref, (curr_k_slice, slice(None)))
        kv_attention_mask = (
            None
            if attention_mask_ref is None
            else pl.load(attention_mask_ref, (curr_k_slice,))
        )
        qk = pl.dot(q, k.T)
        if sm_scale != 1.:
            qk *= sm_scale

        if causal or attention_mask_ref is not None:
            mask = None
            if attention_mask_ref is not None:
                mask = segment_mask(q_attention_mask, kv_attention_mask)
            if causal:
                span_q = start_q * block_q + jnp.arange(block_q)
                span_k = start_k * block_k + jnp.arange(block_k)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )
            # Apply mask to qk.
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        m_curr = qk.max(axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp(qk - m_next[:, None])
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        l_next_rcp = 1. / l_next
        s_curr = s_curr * l_next_rcp[:, None]
        o_prev_corr = (l_prev_corr * l_next_rcp)[:, None] * o_prev
        v = pl.load(v_ref, (curr_k_slice, pl.dslice(block_d)))
        o_curr = pl.dot(s_curr.astype(v.dtype), v)

        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    if causal:
        upper_bound = lax.div(block_q * (start_q + 1) + block_k - 1, block_k)
    else:
        upper_bound = pl.cdiv(seq_len, block_k)
    o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

    if residual_refs:
        l_ref, m_ref = residual_refs
        pl.store(l_ref, (curr_q_slice,), l_i)
        pl.store(m_ref, (curr_q_slice,), m_i)
    # Write output to dram.
    o = o.astype(o_ref.dtype)
    pl.store(o_ref, (curr_q_slice, pl.dslice(None)), o)


def segment_mask(
        q_attention_mask: jax.Array,
        kv_attention_mask: jax.Array,
):
    # [B, T, 1] or [T, 1]
    q_attention_mask = jnp.expand_dims(q_attention_mask, axis=-1)
    # [B, 1, S] or [1, S]
    if kv_attention_mask.ndim == 1:
        kv_attention_mask = jnp.expand_dims(kv_attention_mask, axis=0)
    else:
        kv_attention_mask = jnp.expand_dims(kv_attention_mask, axis=1)
    return jnp.equal(q_attention_mask, kv_attention_mask).astype(jnp.bool_)


@functools.partial(
    jax.custom_vjp, nondiff_argnums=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "block_q",
        "block_k",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
    ],
)
def flash_attention(
        query,
        key,
        value,
        attention_mask: jnp.ndarray | None,
        sm_scale: float = 1.0,
        causal: bool = False,
        block_q: int = 128,
        block_k: int = 128,
        backward_pass_impl: str = "triton",
        num_warps: int | None = None,
        num_stages: int = 2,
        grid: tuple[int, ...] | None = None,
        interpret: bool = ...,
        debug: bool = False,
):
    del backward_pass_impl
    batch_size, seq_len, num_heads, head_dim = query.shape
    block_q = min(block_q, seq_len)
    block_k = min(block_k, seq_len)
    # Heuristics.
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(seq_len, block_q), batch_size, num_heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    kernel = functools.partial(flash_attention_forward_kernel, num_heads=num_heads,
                               sm_scale=sm_scale, block_q=block_q,
                               block_k=block_k, block_d=head_dim,
                               causal=causal)

    in_specs = [
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        None if attention_mask is None else pl.BlockSpec(lambda _, j, k: (j, 0), (None, seq_len))
    ]
    out_shape = jax.ShapeDtypeStruct(shape=query.shape, dtype=query.dtype)
    if interpret == Ellipsis:
        interpret = not (seq_len / 16).is_integer() or jax.lib.xla_bridge.get_backend().platform == "cpu"
    return pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=pl.BlockSpec(
            lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
        ),
        compiler_params=dict(
            triton=dict(num_warps=num_warps_, num_stages=num_stages)
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="flash_attention_forward",
    )(query, key, value, attention_mask)


def _flash_attention_forward(
        q,
        k,
        v,
        attention_mask: jax.Array | None,
        sm_scale: float,
        causal: bool,
        block_q: int,
        block_k: int,
        backward_pass_impl: str,
        num_warps: int | None,
        num_stages: int,
        grid: Any,
        interpret: bool,
        debug: bool,
):
    del backward_pass_impl
    batch_size, seq_len, num_heads, head_dim = q.shape
    block_q = min(block_q, seq_len)
    block_k = min(block_k, seq_len)
    # Heuristics.
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(seq_len, block_q), batch_size, num_heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    kernel = functools.partial(flash_attention_forward_kernel, num_heads=num_heads,
                               sm_scale=sm_scale, causal=causal, block_q=block_q,
                               block_k=block_k, block_d=head_dim)
    out_shape = [
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),  # out
        jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len),  # l
                             dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len),  # m
                             dtype=jnp.float32)
    ]
    in_specs = [
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        None if attention_mask is None else pl.BlockSpec(lambda _, j, k: (j, 0), (None, seq_len))
    ]
    out_specs = [
        pl.BlockSpec(
            lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
        ),
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
    ]
    out, l, m = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=out_specs,  # type:ignore
        compiler_params=dict(
            triton=dict(num_warps=num_warps_, num_stages=num_stages)
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="flash_attention_forward",
    )(q, k, v, attention_mask)
    return out, (q, k, v, attention_mask, out, l, m)


def _preprocess_backward_kernel(
        out_ref,
        d_out_ref,
        l_ref,
        new_d_out_ref,
        delta_ref, *,
        block_q: int
):
    pid_m = pl.program_id(0)

    off_m = pl.ds(pid_m * block_q, block_q)
    # load
    o = pl.load(out_ref, (off_m, slice(None))).astype(jnp.float32)
    do = pl.load(d_out_ref, (off_m, slice(None))).astype(jnp.float32)
    de_num = pl.load(l_ref, (off_m,)).astype(jnp.float32)
    # compute
    do = do / de_num[:, None]
    delta = jnp.sum(o * do, axis=1)
    # write-back
    pl.store(new_d_out_ref, (off_m, slice(None)), do.astype(new_d_out_ref.dtype))
    pl.store(delta_ref, (off_m,), delta.astype(delta_ref.dtype))


@jax.named_scope("preprocess_backward")
def _preprocess_backward(
        out,
        do,
        l,
        block_q: int,
        debug: bool,
        interpret: bool
):
    batch_size, seq_len, num_heads, head_dim = out.shape
    out_shape = [
        jax.ShapeDtypeStruct(do.shape, do.dtype),
        jax.ShapeDtypeStruct(l.shape, l.dtype),
    ]
    out_specs = [
        pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
    ]
    do_scaled, delta = pl.pallas_call(
        functools.partial(_preprocess_backward_kernel, block_q=block_q),
        grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
        in_specs=[
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        ],
        out_specs=out_specs,  # type:ignore
        compiler_params=dict(
            triton=dict(num_warps=4, num_stages=3)
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="flash_attention_preprocess_backward")(out, do, l)
    return do_scaled, delta


def flash_attention_backward_kernel(
        # Inputs
        q_ref,
        k_ref,
        v_ref,
        attention_mask_ref: jax.Array | None,
        out_ref,
        do_scaled_ref,
        l_ref,
        m_ref,
        delta_ref,
        _,
        # Outputs
        dq_ref,
        dk_ref,
        dv_ref,
        *,
        sm_scale: float,
        causal: bool,
        block_q: int,
        block_d: int,
        block_k: int,
):
    del out_ref, l_ref  # Not needed
    seq_len = q_ref.shape[0]

    def outer_loop(start_k, _):

        dv = jnp.zeros([block_k, block_d], dtype=jnp.float32)
        dk = jnp.zeros([block_k, block_d], dtype=jnp.float32)
        k = pl.load(k_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
        v = pl.load(v_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
        span_k = start_k * block_k + jnp.arange(block_k)
        kv_attention_mask = (
            None
            if attention_mask_ref is None
            else pl.load(attention_mask_ref, (pl.ds(start_k * block_k, block_k),))
        )

        def inner_loop(start_q, carry):
            dv, dk = carry
            q = pl.load(q_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
            qk = pl.dot(q, k.T)
            qk = qk.astype(q_ref.dtype)
            qk = qk.astype(jnp.float32)
            if sm_scale != 1.0:
                qk *= sm_scale

            q_attention_mask = (
                None
                if attention_mask_ref is None
                else pl.load(attention_mask_ref, (pl.ds(start_q * block_q, block_q),))
            )

            if causal or attention_mask_ref is not None:
                mask = None
                if attention_mask_ref is not None:
                    mask = segment_mask(q_attention_mask, kv_attention_mask)

                if causal:
                    span_q = start_q * block_q + jnp.arange(block_q)
                    causal_mask = span_q[:, None] >= span_k[None, :]
                    mask = (
                        causal_mask
                        if mask is None
                        else jnp.logical_and(mask, causal_mask)
                    )
                qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

            m = pl.load(m_ref, (pl.ds(start_q * block_q, block_q),))
            p = jnp.exp(qk - m[:, None])
            do = pl.load(do_scaled_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
            dv = dv + pl.dot(p.astype(do.dtype).T, do)
            di = pl.load(delta_ref, (pl.ds(start_q * block_q, block_q),))
            dp = jnp.zeros((block_q, block_k), dtype=jnp.float32) - di[:, None]
            dp = dp + pl.dot(do, v.T)
            ds = p * dp
            if sm_scale != 1.0:
                ds = ds * sm_scale
            dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q)
            dq = pl.load(dq_ref, (pl.ds(start_q * block_q, block_q), slice(None)), eviction_policy="evict_last")
            dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)
            pl.store(dq_ref, (pl.ds(start_q * block_q, block_q), slice(None)), dq, eviction_policy="evict_last")
            return dv, dk

        if causal:
            lower_bound = lax.div(start_k * block_k, block_q)
        else:
            lower_bound = 0
        dv, dk = lax.fori_loop(lower_bound, pl.cdiv(seq_len, block_q), inner_loop, (dv, dk))
        pl.store(dv_ref, (pl.ds(start_k * block_k, block_k), slice(None)), dv.astype(dv_ref.dtype))
        pl.store(dk_ref, (pl.ds(start_k * block_k, block_k), slice(None)), dk.astype(dk_ref.dtype))

    lax.fori_loop(0, pl.cdiv(seq_len, block_k), outer_loop, None)


@functools.partial(jax.jit, static_argnames=["sm_scale", "causal"])
def _flash_attention_reference(
        q,
        k,
        v,
        attention_mask: jnp.ndarray | None,
        sm_scale=1.0,
        causal: bool = False,
):
    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[1]
    logits = jnp.einsum("bqhc,bkhc->bhqk", q, k).astype(jnp.float32)
    mask = None
    if attention_mask is not None:
        mask = jnp.expand_dims(segment_mask(attention_mask, attention_mask), 1)
        mask = jnp.broadcast_to(mask, logits.shape)
    if causal:
        causal_mask = jnp.tril(jnp.ones((1, 1, q_seq_len, kv_seq_len), dtype=bool))
        causal_mask = jnp.broadcast_to(causal_mask, logits.shape)
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))
    weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
    return jnp.einsum("bhqk,bkhc->bqhc", weights, v)


def _flash_attention_backward(
        sm_scale: float,
        causal: bool,
        block_q: int,
        block_k: int,
        backward_pass_impl: str,
        num_warps: int | None,
        num_stages: int,
        grid: Any,
        interpret: bool,
        debug: bool,
        res,
        do
):
    del num_warps, num_stages, grid
    q, k, v, attention_mask, out, l, m = res

    if backward_pass_impl == "xla":
        return jax.vjp(
            functools.partial(_flash_attention_reference, sm_scale=sm_scale, causal=causal),
            q,
            k,
            v,
            attention_mask,
        )[1](do)
    elif backward_pass_impl == "triton":
        batch_size, seq_len, num_heads, head_dim = q.shape
        block_q = min(block_q, seq_len)
        block_k = min(block_k, seq_len)
        do_scaled, delta = _preprocess_backward(out, do, l, block_q, debug, interpret)
        # We accumulate into dq so we need to initialize it to zeros.
        dq = jnp.zeros(q.shape, jnp.float32)
        out_shapes = [
            jax.ShapeDtypeStruct(dq.shape, dq.dtype),
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ]

        in_specs = [
            pl.BlockSpec(
                lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(
                lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(
                lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(
                lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(
                lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
            pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
            pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
            pl.BlockSpec(
                lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
            ),
        ]
        if attention_mask is None:
            in_specs.insert(3, None)  # type: ignore[arg-type]
            input_output_aliases = {8: 0}
        else:
            in_specs.insert(3, pl.BlockSpec(lambda j, k: (j, 0), (None, seq_len)))
            input_output_aliases = {9: 0}
        grid = (batch_size, num_heads)
        num_warps = 8
        out_specs = [
            pl.BlockSpec(
                lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(
                lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(
                lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
            ),
        ]
        dq, dk, dv = pl.pallas_call(
            functools.partial(
                flash_attention_backward_kernel,
                block_q=block_q,
                block_d=head_dim,
                block_k=block_k,
                sm_scale=sm_scale,
                causal=causal,
            ),
            out_specs=out_specs,  # type:ignore
            grid=grid,
            out_shape=out_shapes,
            in_specs=in_specs,
            name="flash_attention_backward",
            debug=debug,
            interpret=interpret,
            compiler_params=dict(
                triton=dict(
                    num_warps=num_warps, num_stages=1
                )
            ),
            input_output_aliases=input_output_aliases,
        )(q, k, v, attention_mask, out, do_scaled, l, m, delta, dq)
    else:
        raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
    return dq.astype(q.dtype), dk, dv, None


flash_attention.defvjp(_flash_attention_forward, _flash_attention_backward)
