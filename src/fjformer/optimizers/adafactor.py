from typing import Any, Optional, Tuple

import chex
import jax.numpy as jnp
import optax

from fjformer.optimizers.optimizer_utils import optax_add_scheduled_weight_decay
import warnings


def _get_adafactor_base(
    scheduler: optax.Schedule,
    min_dim_size_to_factor: int = 128,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: bool = True,
    clipping_threshold: Optional[float] = 1.0,
    momentum: Optional[float] = None,
    dtype_momentum: chex.ArrayDType = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-30,
    factored: bool = True,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Any] = None,
    gradient_accumulation_steps: int = 1,
    clip_grad: Optional[float] = None,
    **kwargs
) -> optax.GradientTransformation:
    """
    Creates a base Adafactor optimizer with the given scheduler and options.

    Args:
        scheduler (optax.Schedule): Learning rate scheduler.
        min_dim_size_to_factor (int): Minimum dimension size for factoring.
        decay_rate (float): Decay rate for moment estimates.
        decay_offset (int): Offset for decay calculations.
        multiply_by_parameter_scale (bool): Whether to scale updates by parameter scale.
        clipping_threshold (Optional[float]): Gradient clipping threshold.
        momentum (Optional[float]): Momentum factor.
        dtype_momentum (chex.ArrayDType): Data type for momentum.
        weight_decay_rate (Optional[float]): Weight decay rate for Adafactor.
        eps (float): Epsilon for numerical stability.
        factored (bool): Whether to use factored second moment estimates.
        weight_decay (float): Additional weight decay factor.
        weight_decay_mask (Optional[Any]): Mask for weight decay.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        optax.GradientTransformation: The configured optimizer.
    """

    for kwarg in kwargs.keys():
        warnings.warn(f"Key {kwarg} is not used for optimizer.")
    chain = [
        optax.adafactor(
            learning_rate=scheduler,
            min_dim_size_to_factor=min_dim_size_to_factor,
            decay_rate=decay_rate,
            decay_offset=decay_offset,
            multiply_by_parameter_scale=multiply_by_parameter_scale,
            clipping_threshold=clipping_threshold,
            eps=eps,
            momentum=momentum,
            weight_decay_rate=weight_decay_rate,
            dtype_momentum=dtype_momentum,
            factored=factored,
        )
    ]
    if clip_grad is not None:
        chain.insert(0, optax.clip_by_global_norm(clip_grad))

    if weight_decay != 0.0:
        chain.append(
            optax_add_scheduled_weight_decay(
                lambda step: -scheduler(step) * weight_decay, weight_decay_mask
            )
        )

    tx = optax.chain(*chain)

    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, gradient_accumulation_steps)

    return tx


def get_adafactor_with_linear_scheduler(
    steps: int,
    learning_rate_start: float = 5e-5,
    learning_rate_end: float = 1e-5,
    weight_decay: float = 1e-1,
    min_dim_size_to_factor: int = 128,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: bool = True,
    clipping_threshold: Optional[float] = 1.0,
    momentum: Optional[float] = None,
    dtype_momentum: chex.ArrayDType = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-30,
    factored: bool = True,
    gradient_accumulation_steps: int = 1,
    weight_decay_mask: Optional[Any] = None,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates an Adafactor optimizer with a linear learning rate scheduler.

    Args:
        steps (int): Total number of training steps.
        learning_rate_start (float): Initial learning rate.
        learning_rate_end (float): Final learning rate.
        weight_decay (float): Weight decay factor.
        min_dim_size_to_factor (int): Minimum dimension size for factoring.
        decay_rate (float): Decay rate for moment estimates.
        decay_offset (int): Offset for decay calculations.
        multiply_by_parameter_scale (bool): Whether to scale updates by parameter scale.
        clipping_threshold (Optional[float]): Gradient clipping threshold.
        momentum (Optional[float]): Momentum factor.
        dtype_momentum (chex.ArrayDType): Data type for momentum.
        weight_decay_rate (Optional[float]): Weight decay rate for Adafactor.
        eps (float): Epsilon for numerical stability.
        factored (bool): Whether to use factored second moment estimates.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        weight_decay_mask (Optional[Any]): Mask for weight decay.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        Tuple[optax.GradientTransformation, optax.Schedule]: The optimizer and scheduler.
    """
    scheduler = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps,
    )

    tx = _get_adafactor_base(
        scheduler=scheduler,
        min_dim_size_to_factor=min_dim_size_to_factor,
        decay_rate=decay_rate,
        decay_offset=decay_offset,
        multiply_by_parameter_scale=multiply_by_parameter_scale,
        clipping_threshold=clipping_threshold,
        momentum=momentum,
        dtype_momentum=dtype_momentum,
        weight_decay_rate=weight_decay_rate,
        eps=eps,
        factored=factored,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        **kwargs,
    )

    return tx, scheduler


def get_adafactor_with_warmup_linear_scheduler(
    steps: int,
    learning_rate_start: float = 5e-5,
    learning_rate_end: float = 1e-5,
    warmup_steps: int = 100,
    min_dim_size_to_factor: int = 128,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: bool = True,
    clipping_threshold: Optional[float] = 1.0,
    momentum: Optional[float] = None,
    dtype_momentum: chex.ArrayDType = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-30,
    factored: bool = True,
    gradient_accumulation_steps: int = 1,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates an Adafactor optimizer with a warm-up linear learning rate scheduler.

    Args:
        steps (int): Total number of training steps.
        learning_rate_start (float): Learning rate after warm-up.
        learning_rate_end (float): Final learning rate.
        warmup_steps (int): Number of warm-up steps.
        min_dim_size_to_factor (int): Minimum dimension size for factoring.
        decay_rate (float): Decay rate for moment estimates.
        decay_offset (int): Offset for decay calculations.
        multiply_by_parameter_scale (bool): Whether to scale updates by parameter scale.
        clipping_threshold (Optional[float]): Gradient clipping threshold.
        momentum (Optional[float]): Momentum factor.
        dtype_momentum (chex.ArrayDType): Data type for momentum.
        weight_decay_rate (Optional[float]): Weight decay rate for Adafactor.
        eps (float): Epsilon for numerical stability.
        factored (bool): Whether to use factored second moment estimates.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        Tuple[optax.GradientTransformation, optax.Schedule]: The optimizer and scheduler.
    """
    scheduler_warmup = optax.linear_schedule(
        init_value=5e-8, end_value=learning_rate_start, transition_steps=warmup_steps
    )
    scheduler_decay = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps - warmup_steps,
    )

    scheduler_combined = optax.join_schedules(
        schedules=[scheduler_warmup, scheduler_decay], boundaries=[warmup_steps]
    )

    tx = _get_adafactor_base(
        scheduler=scheduler_combined,
        min_dim_size_to_factor=min_dim_size_to_factor,
        decay_rate=decay_rate,
        decay_offset=decay_offset,
        multiply_by_parameter_scale=multiply_by_parameter_scale,
        clipping_threshold=clipping_threshold,
        momentum=momentum,
        dtype_momentum=dtype_momentum,
        weight_decay_rate=weight_decay_rate,
        eps=eps,
        factored=factored,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        **kwargs,
    )

    return tx, scheduler_combined


def get_adafactor_with_cosine_scheduler(
    steps: int,
    learning_rate: float = 5e-5,
    min_dim_size_to_factor: int = 128,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: bool = True,
    clipping_threshold: Optional[float] = 1.0,
    momentum: Optional[float] = None,
    dtype_momentum: chex.ArrayDType = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-30,
    factored: bool = True,
    gradient_accumulation_steps: int = 1,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates an Adafactor optimizer with a cosine learning rate scheduler.

    Args:
        steps (int): Total number of training steps.
        learning_rate (float): Peak learning rate.
        min_dim_size_to_factor (int): Minimum dimension size for factoring.
        decay_rate (float): Decay rate for moment estimates.
        decay_offset (int): Offset for decay calculations.
        multiply_by_parameter_scale (bool): Whether to scale updates by parameter scale.
        clipping_threshold (Optional[float]): Gradient clipping threshold.
        momentum (Optional[float]): Momentum factor.
        dtype_momentum (chex.ArrayDType): Data type for momentum.
        weight_decay_rate (Optional[float]): Weight decay rate for Adafactor.
        eps (float): Epsilon for numerical stability.
        factored (bool): Whether to use factored second moment estimates.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        Tuple[optax.GradientTransformation, optax.Schedule]: The optimizer and scheduler.
    """
    scheduler = optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=steps)

    tx = _get_adafactor_base(
        scheduler=scheduler,
        min_dim_size_to_factor=min_dim_size_to_factor,
        decay_rate=decay_rate,
        decay_offset=decay_offset,
        multiply_by_parameter_scale=multiply_by_parameter_scale,
        clipping_threshold=clipping_threshold,
        momentum=momentum,
        dtype_momentum=dtype_momentum,
        weight_decay_rate=weight_decay_rate,
        eps=eps,
        factored=factored,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        **kwargs,
    )

    return tx, scheduler


def get_adafactor_with_warmup_cosine_scheduler(
    steps: int,
    learning_rate: float = 5e-5,
    learning_rate_end: float = 1e-5,
    weight_decay: float = 1e-1,
    min_dim_size_to_factor: int = 128,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: bool = True,
    clipping_threshold: Optional[float] = 1.0,
    momentum: Optional[float] = None,
    dtype_momentum: chex.ArrayDType = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-30,
    factored: bool = True,
    exponent: float = 1.0,
    weight_decay_mask: Optional[Any] = None,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 100,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates an Adafactor optimizer with a warm-up cosine learning rate scheduler.

    Args:
        steps (int): Total number of training steps.
        learning_rate (float): Peak learning rate after warm-up.
        learning_rate_end (float): Final learning rate.
        weight_decay (float): Weight decay factor.
        min_dim_size_to_factor (int): Minimum dimension size for factoring.
        decay_rate (float): Decay rate for moment estimates.
        decay_offset (int): Offset for decay calculations.
        multiply_by_parameter_scale (bool): Whether to scale updates by parameter scale.
        clipping_threshold (Optional[float]): Gradient clipping threshold.
        momentum (Optional[float]): Momentum factor.
        dtype_momentum (chex.ArrayDType): Data type for momentum.
        weight_decay_rate (Optional[float]): Weight decay rate for Adafactor.
        eps (float): Epsilon for numerical stability.
        factored (bool): Whether to use factored second moment estimates.
        exponent (float): Exponent for the cosine decay.
        weight_decay_mask (Optional[Any]): Mask for weight decay.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        warmup_steps (int): Number of warm-up steps.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        Tuple[optax.GradientTransformation, optax.Schedule]: The optimizer and scheduler.
    """
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.5e-7,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=steps,
        end_value=learning_rate_end,
        exponent=exponent,
    )

    tx = _get_adafactor_base(
        scheduler=scheduler,
        min_dim_size_to_factor=min_dim_size_to_factor,
        decay_rate=decay_rate,
        decay_offset=decay_offset,
        multiply_by_parameter_scale=multiply_by_parameter_scale,
        clipping_threshold=clipping_threshold,
        momentum=momentum,
        dtype_momentum=dtype_momentum,
        weight_decay_rate=weight_decay_rate,
        eps=eps,
        factored=factored,
        weight_decay=weight_decay,
        weight_decay_mask=weight_decay_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        **kwargs,
    )

    return tx, scheduler
