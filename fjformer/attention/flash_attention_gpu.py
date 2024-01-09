# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing fused attention forward and backward pass."""
from __future__ import annotations

import functools
from typing import Any, Optional

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def mha_forward_kernel(
        q_ref,
        k_ref,
        v_ref,
        acc_ref,
        l_ref,
        m_ref,  # Input arrays
        q_chunk_idx_start_ref,
        k_chunk_idx_start_ref,
        segment_ids_ref: jax.Array | None,  # segment_id arrays
        o_ref: Any,  # Output
        *residual_refs: Any,  # Residual outputs
        num_heads: int,
        sm_scale: float,
        causal: bool,
        block_q: int,
        block_d: int,
        block_k: int,
):
    """
    The mha_forward_kernel function is the main function that performs the matrix multiplication
    between query and key, as well as between softmax(query*key) and value. It also handles masking
    and residual connections. The kernel is called by mha_forward_kernel_pipeline, which sets up a
    pipeline to run this kernel on multiple blocks of q in parallel.

    :param q_ref: Load the queries from dram to sram
    :param k_ref: Store the key values
    :param v_ref: Load the values from dram
    :param acc_ref: Store the output of the attention layer
    :param l_ref: Store the l_i values
    :param m_ref: Store the max values of qk
    :param q_chunk_idx_start_ref: Store the start index of q_ref
    :param k_chunk_idx_start_ref: Index the k_ref array
    :param segment_ids_ref: jax.Array | None: Determine whether to use segment masking
    :param o_ref: Any: Store the output of the mha_forward_kernel function
    :param # Output
            *residual_refs: Any: Store the residual outputs
    :param # Residual outputs
            num_heads: int: Determine the number of heads in the attention layer
    :param sm_scale: float: Scale the dot product of q and k
    :param causal: bool: Determine whether to use the causal mask or not
    :param block_q: int: Determine the size of the q tile
    :param block_d: int: Specify the size of the output
    :param block_k: int: Determine the size of the kv tile
    :param : Determine the size of the output
    :return: A program
    
    """
    seq_len = q_ref.shape[0]
    start_q = pl.program_id(0)

    # o is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
    m_i = pl.load(m_ref, (pl.dslice(start_q * block_q, block_q),))
    l_i = pl.load(l_ref, (pl.dslice(start_q * block_q, block_q),))
    # acc is the buffer where we accumulate the output on sram.
    o = pl.load(acc_ref, (pl.dslice(start_q * block_q, block_q), pl.dslice(None)))

    q_chunk_idx_start = pl.load(q_chunk_idx_start_ref, (0,))
    k_chunk_idx_start = pl.load(k_chunk_idx_start_ref, (0,))

    # Load q: it will stay in L1 throughout. Indices form a matrix because we
    # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
    # q tile has shape [block_q, block_d], block_d == head_dim.
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = pl.load(q_ref, (curr_q_slice, pl.dslice(None)))
    q_segment_ids = (
        None
        if segment_ids_ref is None
        else pl.load(segment_ids_ref, (curr_q_slice,))
    )

    # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
    # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
    # Here we only loop over blocks of kv to process entire seq_len, the loop over
    # blocks of q is carried out by the grid.
    def body(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pl.load(k_ref, (curr_k_slice, slice(None)))
        kv_segment_ids = (
            None
            if segment_ids_ref is None
            else pl.load(segment_ids_ref, (curr_k_slice,))
        )
        qk = pl.dot(q, k.T)  # [block_q, block_k]
        if sm_scale != 1.:
            qk *= sm_scale  # [block_q, block_k]

        # Avoids Triton crash.
        # if num_heads > 2:
        #   qk = qk.astype(q_ref.dtype)
        #   qk = qk.astype(jnp.float32)

        if causal or segment_ids_ref is not None:
            mask = None
            if segment_ids_ref is not None:
                mask = segment_mask(q_segment_ids, kv_segment_ids)
            if causal:
                span_q = (q_chunk_idx_start + start_q) * block_q + jnp.arange(block_q)
                span_k = (k_chunk_idx_start + start_k) * block_k + jnp.arange(block_k)
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
        s_curr = jnp.exp(
            qk - m_next[:, None]
        )  # Use m_next instead of m_curr to avoid a correction on l_curr
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
        # Ceildiv (`pl.cdiv` and `//` do not work due to type of start_q)
        upper_bound = lax.min(
            lax.div(block_q * (q_chunk_idx_start + start_q + 1) + block_k - 1, block_k) - k_chunk_idx_start,
            pl.cdiv(seq_len, block_k))
    else:
        upper_bound = pl.cdiv(seq_len, block_k)  # type: ignore
    o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

    if residual_refs:
        l_ref, m_ref = residual_refs
        pl.store(l_ref, (curr_q_slice,), l_i)
        pl.store(m_ref, (curr_q_slice,), m_i)
    # Write output to dram.
    o = o.astype(o_ref.dtype)
    pl.store(o_ref, (curr_q_slice, pl.dslice(None)), o)


def segment_mask(
        q_segment_ids: jax.Array,
        kv_segment_ids: jax.Array,
):
    # [B, T, 1] or [T, 1]
    """
    The segment_mask function is used to mask out the attention scores for
    the query-key pairs that are not in the same segment. This is done by
    creating a boolean array of shape [B, T, S] or [T, S], where B and T are
    the batch size and sequence length of the query tensor respectively; and S
    is the sequence length of key/value tensors. The boolean array has True at
    positions where q_segment_ids == kv_segment_ids (i.e., when they belong to
    the same segment), otherwise False.

    :param q_segment_ids: jax.Array: Create a mask for the query
    :param kv_segment_ids: jax.Array: Create a mask
    :param : Mask the attention weights
    :return: A boolean mask that is true for all the
    
    """
    q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
    # [B, 1, S] or [1, S]
    if kv_segment_ids.ndim == 1:
        kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=0)
    else:
        kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=1)
    return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)


def _mha_forward(
        q,
        k,
        v,
        carry,
        q_chunk_idx_start,
        k_chunk_idx_start,
        segment_ids: jax.Array | None,
        sm_scale: float,
        causal: bool,
        block_q: int,
        block_k: int,
        backward_pass_impl: str,
        num_warps: Optional[int],
        num_stages: int,
        grid: Any,
        interpret: bool,
        debug: bool,
):
    """
    The _mha_forward function is a wrapper for the mha_forward_kernel function.
    It takes in the query, key, and value tensors as well as other parameters such
    as segment ids (if applicable), sm scale (scaling factor for softmax), causal
    (whether or not to use causal attention), block q/k (the size of each chunk of
    the sequence length dimension that will be processed by one thread block). It also takes in num warps which is the number of warps per SM on GPU. The grid parameter specifies how many blocks are launched per SM. If it's None then we use heur

    :param q: Store the input query
    :param k: Compute the attention weights
    :param v: Store the output of the attention
    :param carry: Store the output of the previous iteration
    :param q_chunk_idx_start: Keep track of the current chunk
    :param k_chunk_idx_start: Determine the starting index of the key
    :param segment_ids: jax.Array | None: Determine whether the
    :param sm_scale: float: Scale the softmax function
    :param causal: bool: Determine whether the model is causal or not
    :param block_q: int: Specify the number of blocks in the query
    :param block_k: int: Determine the number of blocks in the kernel
    :param backward_pass_impl: str: Choose the implementation of the backward pass
    :param num_warps: Optional[int]: Set the number of warps
    :param num_stages: int: Control the number of stages in the pipeline
    :param grid: Any: Specify the size of the grid
    :param interpret: bool: Print out the kernel code
    :param debug: bool: Print out the shape of each input and output
    :param : Determine the number of heads
    :return: The following:
    
    """
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
    kernel = functools.partial(mha_forward_kernel, num_heads=num_heads,
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
        pl.BlockSpec(
            lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
        ),  # q
        pl.BlockSpec(
            lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
        ),  # k
        pl.BlockSpec(
            lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
        ),  # v
        pl.BlockSpec(
            lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
        ),  # acc
        pl.BlockSpec(
            lambda _, j, k: (j, k, 0), (None, None, seq_len)
        ),  # l
        pl.BlockSpec(
            lambda _, j, k: (j, k, 0), (None, None, seq_len)
        ),  # m
    ]
    in_specs.append(pl.BlockSpec(lambda _, j, k: (0,), (1,)))
    in_specs.append(pl.BlockSpec(lambda _, j, k: (0,), (1,)))
    in_specs.append(
        None  # type: ignore[arg-type]
        if segment_ids is None
        else pl.BlockSpec(lambda _, j, k: (j, 0), (None, seq_len))
    )
    out, l, m = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=[
            pl.BlockSpec(
                lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
            ),
            pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
            pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        ],
        num_warps=num_warps_,
        num_stages=num_stages,
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k, v, *carry, q_chunk_idx_start[None], k_chunk_idx_start[None], segment_ids)
    return out, l, m


def _preprocess_backward_kernel(out_ref, dout_ref, l_ref,
                                new_dout_ref, delta_ref, *,
                                block_q: int):
    pid_m = pl.program_id(0)

    off_m = pl.ds(pid_m * block_q, block_q)
    # checkpoint
    o = pl.load(out_ref, (off_m, slice(None))).astype(jnp.float32)
    do = pl.load(dout_ref, (off_m, slice(None))).astype(jnp.float32)
    denom = pl.load(l_ref, (off_m,)).astype(jnp.float32)
    # compute
    do = do / denom[:, None]
    delta = jnp.sum(o * do, axis=1)
    # write-back
    pl.store(new_dout_ref, (off_m, slice(None)),
             do.astype(new_dout_ref.dtype))
    pl.store(delta_ref, (off_m,), delta.astype(delta_ref.dtype))


@jax.named_scope("preprocess_backward")
def _preprocess_backward(out, do, l, block_q: int,
                         debug: bool, interpret: bool):
    """
    The _preprocess_backward function is the backward pass of the preprocess function.
    It takes in a tensor ``out``, which is the output of _preprocess_forward, and two other
    tensors ``do`` and ``l``. The first one represents a scaled version of the attention weights
    and has shape (batch_size, seq_len, num_heads). The second one represents an intermediate value used to compute
    the gradient with respect to ln(q) and has shape (batch_size * num heads). It returns two tensors: do' which
    represents a scaled version of datt

    :param out: Compute the do_scaled parameter
    :param do: Store the output of the previous layer
    :param l: Store the length of each sequence in a batch
    :param block_q: int: Control the number of threads in a block
    :param debug: bool: Enable the debug mode of pallas
    :param interpret: bool: Enable the interpreter mode
    :return: A tuple of two arrays
    
    """
    batch_size, seq_len, num_heads, head_dim = out.shape
    out_shape = [
        jax.ShapeDtypeStruct(do.shape, do.dtype),
        jax.ShapeDtypeStruct(l.shape, l.dtype),
    ]
    do_scaled, delta = pl.pallas_call(
        functools.partial(_preprocess_backward_kernel, block_q=block_q),
        grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
        in_specs=[
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        ],
        out_specs=[
            pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)),
            pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
        ],
        num_warps=4,
        num_stages=3,
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_preprocess_backward")(out, do, l)
    return do_scaled, delta


def mha_backward_kernel(
        # Inputs
        q_ref,
        k_ref,
        v_ref,
        segment_ids_ref: jax.Array | None,
        out_ref,
        do_scaled_ref,
        l_ref,
        m_ref,
        delta_ref,
        _,
        q_chunk_idx_start_ref,
        k_chunk_idx_start_ref,
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
    """
    The mha_backward_kernel function is the backward pass of the multi-head attention
    module. It takes in a reference to q, k, v and out tensors as well as references to
    the segment_ids (if any), do_scaled (which is dout * scaled), l and m tensors. The
    l and m tensors are used for calculating the softmax gradient. The function also takes in a delta value which is used for calculating gradients when there are masked values present in qk or when causal attention has been applied. Finally it also takes in references to chunk indices that indicate where each chunk starts within its respective sequence

    :param # Inputs
            q_ref: Load the query tensor
    :param k_ref: Load the k_ref array
    :param v_ref: Store the value matrix
    :param segment_ids_ref: jax.Array | None: Determine whether or not to use segment masking
    :param out_ref: Store the output of the mha_forward_kernel function
    :param do_scaled_ref: Store the scaled output of the dot product between q and k
    :param l_ref: Store the logits
    :param m_ref: Store the maximum value of qk
    :param delta_ref: Store the difference between the logits and softmax(logits)
    :param _: Pass in the device
    :param q_chunk_idx_start_ref: Keep track of the start index for each chunk
    :param k_chunk_idx_start_ref: Store the index of the current chunk in k_ref
    :param # Outputs
            dq_ref: Store the output of the function
    :param dk_ref: Store the gradients of k_ref
    :param dv_ref: Store the output of the function
    :param *: Pass in the parameters for the mha_backward_kernel function
    :param sm_scale: float: Scale the qk matrix
    :param causal: bool: Determine whether the mask should be applied
    :param block_q: int: Determine the number of q values to be processed at a time
    :param block_d: int: Specify the dimension of the dv_ref array
    :param block_k: int: Determine the size of the block_k
    :param : Determine the number of blocks in the sequence
    :return: None
    
    """
    del out_ref, l_ref  # Not needed
    seq_len = q_ref.shape[0]
    q_chunk_idx_start = pl.load(q_chunk_idx_start_ref, (0,))
    k_chunk_idx_start = pl.load(k_chunk_idx_start_ref, (0,))

    def outer_loop(start_k, _):

        dv = jnp.zeros([block_k, block_d], dtype=jnp.float32)
        dk = jnp.zeros([block_k, block_d], dtype=jnp.float32)
        k = pl.load(k_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
        v = pl.load(v_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
        span_k = (k_chunk_idx_start + start_k) * block_k + jnp.arange(block_k)
        kv_segment_ids = (
            None
            if segment_ids_ref is None
            else pl.load(segment_ids_ref, (pl.ds(start_k * block_k, block_k),))
        )

        def inner_loop(start_q, carry):
            dv, dk = carry
            q = pl.load(q_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
            qk = pl.dot(q, k.T)
            qk = qk.astype(q_ref.dtype)
            qk = qk.astype(jnp.float32)
            if sm_scale != 1.0:
                qk *= sm_scale

            q_segment_ids = (
                None
                if segment_ids_ref is None
                else pl.load(segment_ids_ref, (pl.ds(start_q * block_q, block_q),))
            )

            if causal or segment_ids_ref is not None:
                mask = None
                if segment_ids_ref is not None:
                    mask = segment_mask(q_segment_ids, kv_segment_ids)

                if causal:
                    span_q = (q_chunk_idx_start + start_q) * block_q + jnp.arange(block_q)
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
            dq = pl.load(dq_ref, (pl.ds(start_q * block_q, block_q),
                                  slice(None)), eviction_policy="evict_last")
            dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)
            pl.store(dq_ref, (pl.ds(start_q * block_q, block_q),
                              slice(None)), dq, eviction_policy="evict_last")
            return dv, dk

        if causal:
            lower_bound = lax.max(0, lax.div((k_chunk_idx_start + start_k) * block_k, block_q) - q_chunk_idx_start)
        else:
            lower_bound = 0
        dv, dk = lax.fori_loop(lower_bound, pl.cdiv(seq_len, block_q), inner_loop,
                               (dv, dk))
        pl.store(dv_ref, (pl.ds(start_k * block_k, block_k),
                          slice(None)), dv.astype(dv_ref.dtype))
        pl.store(dk_ref, (pl.ds(start_k * block_k, block_k),
                          slice(None)), dk.astype(dk_ref.dtype))

    lax.fori_loop(0, pl.cdiv(seq_len, block_k), outer_loop, None)


def _mha_backward(sm_scale: float, causal: bool, block_q: int, block_k: int,
                  backward_pass_impl: str, num_warps: Optional[int],
                  num_stages: int, grid: Any, interpret: bool,
                  debug: bool, q_chunk_idx_start, k_chunk_idx_start, res, do):
    """
    The _mha_backward function is a helper function that computes the backward pass of
    the MHA operation. It takes in the following arguments:
        - sm_scale: The scale factor for softmax. This is used to prevent overflow when computing softmax.
        - causal: Whether to use causal attention (i.e., mask future positions). If True, then we will only attend to past positions and ignore future ones (this is useful for language modeling). If False, then we will attend over all positions in the sequence (this is useful for translation).
        - block_q: The size of each chunk along q

    :param sm_scale: float: Scale the softmax output
    :param causal: bool: Determine whether the mask is causal or not
    :param block_q: int: Specify the number of rows in a block
    :param block_k: int: Define the block size of the key matrix
    :param backward_pass_impl: str: Specify whether to use the
    :param num_warps: Optional[int]: Specify the number of warps
    :param num_stages: int: Control the number of stages in the pallas kernel
    :param grid: Any: Pass the grid to the kernel
    :param interpret: bool: Enable the interpreter mode
    :param debug: bool: Print the intermediate results of the kernel
    :param q_chunk_idx_start: Determine the starting index of q in a chunked version of q
    :param k_chunk_idx_start: Indicate the start index of the current chunk
    :param res: Pass the output of the forward pass to the backward function
    :param do: Accumulate the gradient into dq
    :return: The gradient for the query, key and value
    
    """
    del num_warps, num_stages, grid
    q, k, v, segment_ids, out, l, m = res

    if backward_pass_impl == "xla":
        raise Exception("use backward_pass_impl == triton")
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
        if segment_ids is None:
            in_specs.insert(3, None)  # type: ignore[arg-type]
            input_output_aliases = {8: 0}
        else:
            in_specs.insert(3, pl.BlockSpec(lambda j, k: (j, 0), (None, seq_len)))
            input_output_aliases = {9: 0}
        in_specs.append(pl.BlockSpec(lambda j, k: (0,), (1,)))
        in_specs.append(pl.BlockSpec(lambda j, k: (0,), (1,)))
        grid = (batch_size, num_heads)
        # TODO(sharadmv): figure out why num_warps=8 doesn't work!
        num_warps = 8
        dq, dk, dv = pl.pallas_call(
            functools.partial(
                mha_backward_kernel,
                block_q=block_q,
                block_d=head_dim,
                block_k=block_k,
                sm_scale=sm_scale,
                causal=causal,
            ),
            grid=grid,
            out_shape=out_shapes,
            in_specs=in_specs,
            out_specs=[
                pl.BlockSpec(
                    lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
                ),
                pl.BlockSpec(
                    lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
                ),
                pl.BlockSpec(
                    lambda j, k: (j, 0, k, 0), (None, seq_len, None, head_dim)
                ),
            ],
            name="mha_backward",
            debug=debug,
            interpret=interpret,
            num_warps=num_warps,
            num_stages=1,
            input_output_aliases=input_output_aliases,
        )(q, k, v, segment_ids, out, do_scaled, l, m, delta, dq, q_chunk_idx_start[None], k_chunk_idx_start[None])
    else:
        raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
    return dq.astype(q.dtype), dk, dv
