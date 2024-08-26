import functools
from typing import NamedTuple

import chex
import jax
import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange


class Carry(NamedTuple):
	numerator: chex.Array
	denominator: chex.Array
	max_so_far: chex.Array


def efficient_attention(
	query: chex.Array,
	key: chex.Array,
	value: chex.Array,
	bias: chex.Array = None,
	deterministic: bool = True,
	dropout_rng: chex.PRNGKey = None,
	attention_drop_rate: float = 0.0,
	causal: bool = True,
	query_chunk_size: int = 1024,
	key_chunk_size: int = 1024,
	dtype: chex.ArrayDType = jnp.float32,  # type:ignore
	policy=jax.checkpoint_policies.nothing_saveable(),  # noqa
	precision=None,
	float32_logits: bool = True,
	prevent_cse: bool = True,
):
	"""Memory-efficient attention implementation with optional chunking and causal masking.

	This function implements the multi-head scaled dot-product attention mechanism
	with several efficiency optimizations:

	- **Chunking:**
		- The inputs are split into chunks along the length dimension, and attention
		  is computed one chunk at a time.
		- This reduces memory usage significantly for long sequences, as it avoids
		  computing the entire attention matrix at once.
		- Chunking can be controlled using the `query_chunk_size` and `key_chunk_size`
		  parameters.
	- **Causal masking:**
		- For autoregressive tasks, a causal mask can be applied to prevent the attention
		  mechanism from attending to future tokens.
		- This is enabled by setting the `causal` parameter to `True`.
	- **Dropouts:**
		- Attention dropout is applied to the attention weights before softmax
		  normalization.
		- This can help to prevent overfitting and improve generalization performance.
	- **Checkpoint gradients:**
		- Gradient checkpointing can be used to further reduce memory usage during
		  training.
		- This is enabled by passing a checkpointing policy to the `policy` parameter.
	- **Mixed precision:**
		- The attention logits can be computed in float32 precision, even when the inputs
		  are in a lower precision (e.g., bfloat16).
		- This can improve numerical stability and reduce memory usage.

	Args:
		query: Query matrix with shape `(batch, q_len, num_heads, dim_per_head)`.
		key: Key matrix with shape `(batch, kv_len, num_heads, dim_per_head)`.
		value: Value matrix with shape `(batch, kv_len, num_heads, dim_per_head)`.
		bias: Optional bias matrix with shape broadcastable to `(batch, num_heads, q_len, kv_len)`.
		deterministic: Whether to apply dropout or not.
		dropout_rng: JAX PRNG key for dropout.
		attention_drop_rate: Dropout rate for attention weights.
		causal: Whether to apply a causal mask.
		query_chunk_size: Chunk size for the query matrix.
		key_chunk_size: Chunk size for the key and value matrices.
		dtype: Data type for the output.
		policy: Checkpointing policy for gradient checkpointing.
		precision: Precision for matrix multiplication.
		float32_logits: Whether to compute the attention logits in float32 precision.
		prevent_cse: Whether to prevent common subexpression elimination.

	Returns:
		The attended output with shape `(batch, q_len, num_heads, dim_per_head)`.

	Example usage:

	```python
	# Compute memory-efficient attention with causal masking
	output = efficient_attention(
	  query=query, key=key, value=value, causal=True, query_chunk_size=512
	)

	# Compute memory-efficient attention with dropout and gradient checkpointing
	output = efficient_attention(
	  query=query,
	  key=key,
	  value=value,
	  deterministic=False,
	  dropout_rng=rng,
	  attention_drop_rate=0.1,
	  policy=jax.checkpoint_policies.checkpoint_dots(),
	)
	```
	"""
	query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
	if float32_logits:
		query = query.astype(jnp.float32)
		key = key.astype(jnp.float32)

	batch, q_len, num_heads, dim_per_head = query.shape
	batch, kv_len, kv_heads, dim_per_head = key.shape
	batch, kv_len, kv_heads, dim_per_head = value.shape

	num_q = q_len // query_chunk_size
	num_kv = kv_len // key_chunk_size
	query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
	key = key.reshape((batch, num_kv, key_chunk_size, kv_heads, dim_per_head))
	value = value.reshape((batch, num_kv, key_chunk_size, kv_heads, dim_per_head))

	query = jnp.moveaxis(query, 1, 0)
	key = jnp.moveaxis(key, 1, 0)
	value = jnp.moveaxis(value, 1, 0)

	if bias is not None:
		for bias_dim, broadcast_dim in zip(bias.shape, (batch, num_heads, q_len, kv_len)):  # noqa
			assert bias_dim == 1 or bias_dim == broadcast_dim
	if not deterministic and attention_drop_rate > 0.0:
		attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
		attn_dropout = jax.random.bernoulli(
			attn_dropout_rng, attention_drop_rate, (batch, num_heads, q_len, kv_len)
		)
	else:
		attn_dropout = None

	_chunk_bias_fn = functools.partial(
		_chunk_attention_bias,
		query_chunk_size,
		key_chunk_size,
		bias,
		deterministic,
		attn_dropout,
		attention_drop_rate,
		causal,
		dtype,
	)

	def scan_attention(args):
		query_chunk, query_chunk_idx = args

		@functools.partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
		def scan_kv_block(carry, args):
			key_chunk, value_chunk, key_chunk_idx = args
			(numerator, denominator, prev_max_score) = carry
			attn_weights = jnp.einsum(
				"bqhd,bkhd->bqhk", query_chunk, key_chunk, precision=precision
			)
			bias_chunk = _chunk_bias_fn(query_chunk_idx, key_chunk_idx)
			bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
			attn_weights = attn_weights + bias_chunk

			max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
			max_score = jnp.maximum(prev_max_score, max_score)
			max_score = jax.lax.stop_gradient(max_score)
			exp_weights = jnp.exp(attn_weights - max_score)
			exp_values = jnp.einsum(
				"bqhv,bvhd->bqhd", exp_weights, value_chunk, precision=precision
			)
			correction = jnp.exp(prev_max_score - max_score)
			numerator = numerator * correction + exp_values
			denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)
			return Carry(numerator, denominator, max_score), None

		def skip_upper_half(carry, args):
			key_chunk, value_chunk, key_chunk_idx = args
			skip_block = jnp.array(False)
			if causal:
				skip_block = query_chunk_idx < key_chunk_idx
			return jax.lax.cond(
				skip_block,
				lambda carry, args: (carry, None),
				scan_kv_block,
				carry,
				args,
			)

		init_carry = Carry(
			jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
			jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
			(-jnp.inf) * jnp.ones((batch, query_chunk_size, num_heads, 1), dtype=query.dtype),
		)
		(numerator, denominator, max_score), _ = lax.scan(
			skip_upper_half, init_carry, xs=(key, value, jnp.arange(0, num_kv))
		)
		outputs = (numerator / denominator).astype(dtype)
		return outputs

	_, res = lax.scan(
		lambda _, x: ((), scan_attention(x)), (), xs=(query, jnp.arange(0, num_q))
	)
	res = rearrange(res, "n b c h d -> b (n c) h d")
	return res


def _chunk_attention_bias(
	query_chunk_size: int,
	key_chunk_size: int,
	bias: chex.Array,
	deterministic: bool,
	attn_dropout: chex.Array,
	attention_drop_rate: float,
	causal: bool,
	dtype: chex.ArrayDType,  # type:ignore
	query_chunk_idx: int,
	key_chunk_idx: int,
):
	"""Helper function to compute the attention bias for a specific chunk."""
	query_offset = query_chunk_idx * query_chunk_size
	key_offset = key_chunk_idx * key_chunk_size
	chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
	if bias is not None:
		chunk_bias = lax.dynamic_slice(
			bias,
			start_indices=(0, 0, query_offset, key_offset),
			slice_sizes=(
				*bias.shape[:2],
				min(bias.shape[-2], query_chunk_size),
				min(bias.shape[-1], key_chunk_size),
			),
		)

	if causal:
		query_idx = lax.broadcasted_iota(
			dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0
		)
		key_idx = lax.broadcasted_iota(
			dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1
		)
		offset = query_offset - key_offset
		query_idx += offset
		causal_mask_value = (query_idx < key_idx) * jnp.finfo(dtype).min
		chunk_bias += causal_mask_value.reshape(1, 1, *causal_mask_value.shape)

	if not deterministic and attention_drop_rate > 0.0:
		attn_dropout_slice = lax.dynamic_slice(
			attn_dropout,
			start_indices=(0, 0, query_offset, key_offset),
			slice_sizes=(
				*attn_dropout.shape[:2],
				min(attn_dropout.shape[-2], query_chunk_size),
				min(attn_dropout.shape[-1], key_chunk_size),
			),
		)
		chunk_bias += attn_dropout_slice * jnp.finfo(dtype).min
	return chunk_bias.astype(dtype)
