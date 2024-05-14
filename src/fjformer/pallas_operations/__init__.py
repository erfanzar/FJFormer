"""
# File: __init__.py

## Purpose:
This file contains imports for various attention mechanisms used in our project.
"""

from .efficient_attention import efficient_attention as efficient_attention
from .tpu_flash_attention import (
    flash_attention as tpu_flash_attention,
    mha as gpu_flash_attention,
    BlockSizes
)
from .splash_attention import (
    splash_flash_attention_kernel as splash_flash_attention_kernel,
    SplashAttentionKernel as SplashAttentionKernel,
    BlockSizes as BlockSizes,
    attention_reference as attention_reference,
    Mask as Mask,
    LocalMask as LocalMask,
    make_local_attention_mask as make_local_attention_mask,
    make_causal_mask as make_causal_mask,
    make_random_mask as make_random_mask,
    FullMask as FullMask,
    NumpyMask as NumpyMask,
    CausalMask as CausalMask,
    MultiHeadMask as MultiHeadMask,
    MaskInfo as MaskInfo,
    process_mask as process_mask,
    process_mask_dkv as process_mask_dkv,
    make_splash_mqa_single_device as make_splash_mqa_single_device,
    make_splash_mha_single_device as make_splash_mha_single_device,
    make_splash_mha as make_splash_mha,
    make_splash_mqa as make_splash_mqa,
    make_masked_mqa_reference as make_masked_mqa_reference,
    make_masked_mha_reference as make_masked_mha_reference,
    QKVLayout as QKVLayout,
    SegmentIds as SegmentIds,
    LogicalAnd as LogicalAnd,
    LogicalOr as LogicalOr
)

from .ring_attention import (
    ring_attention_standard as ring_attention_standard,
    ring_flash_attention_tpu as ring_flash_attention_tpu,
    ring_attention as ring_attention
)

from .pallas_flash_attention import flash_attention as flash_attention

__all__ = (

    # Splash Attention
    "MaskInfo",
    "process_mask",
    "process_mask_dkv",
    "Mask",
    "LocalMask",
    "make_local_attention_mask",
    "make_causal_mask",
    "make_random_mask",
    "FullMask",
    "NumpyMask",
    "CausalMask",
    "MultiHeadMask",
    "LogicalOr",
    "LogicalAnd",
    "splash_flash_attention_kernel",
    "SplashAttentionKernel",
    "BlockSizes",
    "attention_reference",
    "SegmentIds",
    "make_splash_mha",
    "make_splash_mqa",
    "make_masked_mha_reference",
    "make_splash_mha_single_device",
    "make_splash_mqa_single_device",
    "make_masked_mqa_reference",
    "QKVLayout",

    # Ring

    "ring_attention_standard",
    "ring_flash_attention_tpu",
    "ring_attention",

    # Flash Attention
    "tpu_flash_attention",
    "gpu_flash_attention",

    # Efficient Attention
    "efficient_attention",
    "flash_attention"
)
