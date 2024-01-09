"""
# File: __init__.py

## Purpose:
This file contains imports for various attention mechanisms used in our project.

## Imports:
1. `efficient_attention` from `efficient_attention` module: This import brings in the `efficient_attention` function,
 which implements an efficient attention mechanism for our project.

2. `dot_product_attention_multihead`, `dot_product_attention_multiquery`, `dot_product_attention_queries_per_head` from
 `flash_attention_0` module: These imports bring in different variations of dot product attention mechanisms for
 multi-head and multi-query scenarios.

3. `ring_attention`, `ring_attention_standard`, `ring_flash_attention_gpu`, `ring_flash_attention_tpu`, `blockwise_ffn`,
 `blockwise_attn` from `flash_attention` module: These imports bring in various attention mechanisms such as
 ring attention, flash attention for GPU and TPU, and blockwise feed-forward network and attention mechanisms.

4. `tpu_flash_attention` `gpu_flash_attention` are imported from jax_flash_attn_tpu/gpu

"""

from .efficient_attention import efficient_attention as efficient_attention
from .flash_attention_0 import (
    dot_product_attention_multihead as dot_product_attention_multihead,
    dot_product_attention_multiquery as dot_product_attention_multiquery,
    dot_product_attention_queries_per_head as dot_product_attention_queries_per_head
)
from .flash_attention import (
    ring_attention as ring_attention,
    ring_attention_standard as ring_attention_standard,
    ring_flash_attention_gpu as ring_flash_attention_gpu,
    ring_flash_attention_tpu as ring_flash_attention_tpu,
    blockwise_ffn as blockwise_ffn,
    blockwise_attn as blockwise_attn
)
from .jax_flash_attn_tpu import flash_attention as tpu_flash_attention, BlockSizes as BlockSizes
from .jax_flash_attn_gpu import mha as gpu_flash_attention
