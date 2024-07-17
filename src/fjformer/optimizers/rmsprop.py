import warnings
from typing import Optional

import optax


def get_rmsprop_with_cosine_scheduler(
    steps: int,
    learning_rate: float = 5e-5,
    decay: float = 0.9,
    initial_scale: float = 0.0,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    eps: float = 1e-8,
    weight_decay: float = 1e-1,
    gradient_accumulation_steps: int = 1,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates an RMSprop optimizer with a cosine learning rate scheduler.

    Args:
        steps: The total number of training steps.
        learning_rate: The initial learning rate.
        decay: The decay rate for the moving average of the squared gradients.
        initial_scale: The initial scale for the moving average of the squared gradients.
        momentum: The momentum value to use. If None, momentum is not used.
        nesterov: Whether to use Nesterov momentum.
        eps: A small value added to the denominator for numerical stability.
        weight_decay: The weight decay rate.
        gradient_accumulation_steps: The number of steps to accumulate gradients over.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        A tuple containing the optimizer and the learning rate scheduler.
    """
    for kwarg in kwargs.keys():
        warnings.warn(f"Key {kwarg} is not used for optimizer.")
    scheduler = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=steps,
    )

    chain = [
        optax.scale_by_rms(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
        (
            optax.trace(
                decay=momentum,
                nesterov=nesterov,
            )
            if momentum is not None
            else optax.identity()
        ),
        optax.add_decayed_weights(
            weight_decay=weight_decay,
        ),
    ]

    if clip_grad is not None:
        chain.insert(0, optax.clip_by_global_norm(clip_grad))
    tx = optax.chain(*chain)
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, gradient_accumulation_steps)
    return tx, scheduler


def get_rmsprop_with_linear_scheduler(
    steps: int,
    learning_rate_start: float = 5e-5,
    learning_rate_end: float = 1e-5,
    decay: float = 0.9,
    initial_scale: float = 0.0,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    eps: float = 1e-8,
    weight_decay: float = 1e-1,
    gradient_accumulation_steps: int = 1,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates an RMSprop optimizer with a linear learning rate scheduler.

    Args:
        steps: The total number of training steps.
        learning_rate_start: The initial learning rate.
        learning_rate_end: The final learning rate.
        decay: The decay rate for the moving average of the squared gradients.
        initial_scale: The initial scale for the moving average of the squared gradients.
        momentum: The momentum value to use. If None, momentum is not used.
        nesterov: Whether to use Nesterov momentum.
        eps: A small value added to the denominator for numerical stability.
        weight_decay: The weight decay rate.
        gradient_accumulation_steps: The number of steps to accumulate gradients over.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        A tuple containing the optimizer and the learning rate scheduler.
    """
    for kwarg in kwargs.keys():
        warnings.warn(f"Key {kwarg} is not used for optimizer.")
    scheduler = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps,
    )

    chain = [
        optax.scale_by_rms(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
        (
            optax.trace(
                decay=momentum,
                nesterov=nesterov,
            )
            if momentum is not None
            else optax.identity()
        ),
        optax.add_decayed_weights(
            weight_decay=weight_decay,
        ),
    ]
    if clip_grad is not None:
        chain.insert(0, optax.clip_by_global_norm(clip_grad))
    tx = optax.chain(*chain)
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, gradient_accumulation_steps)
    return tx, scheduler


def get_rmsprop_with_warmup_linear_scheduler(
    steps: int,
    learning_rate_start: float = 5e-5,
    learning_rate_end: float = 1e-5,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    weight_decay: float = 1e-1,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 100,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates an RMSprop optimizer with a linear learning rate scheduler with warmup.

    Args:
        steps: The total number of training steps.
        learning_rate_start: The initial learning rate after warmup.
        learning_rate_end: The final learning rate.
        decay: The decay rate for the moving average of the squared gradients.
        eps: A small value added to the denominator for numerical stability.
        initial_scale: The initial scale for the moving average of the squared gradients.
        momentum: The momentum value to use. If None, momentum is not used.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: The weight decay rate.
        gradient_accumulation_steps: The number of steps to accumulate gradients over.
        warmup_steps: The number of warmup steps.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        A tuple containing the optimizer and the learning rate scheduler.
    """
    for kwarg in kwargs.keys():
        warnings.warn(f"Key {kwarg} is not used for optimizer.")
    scheduler_warmup = optax.linear_schedule(
        init_value=5e-8,
        end_value=learning_rate_start,
        transition_steps=warmup_steps,
    )
    scheduler_decay = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps - warmup_steps,
    )

    scheduler_combined = optax.join_schedules(
        schedules=[scheduler_warmup, scheduler_decay],
        boundaries=[warmup_steps],
    )

    chain = [
        optax.scale_by_rms(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
        ),
        optax.scale_by_schedule(scheduler_combined),
        optax.scale(-1),
        (
            optax.trace(
                decay=momentum,
                nesterov=nesterov,
            )
            if momentum is not None
            else optax.identity()
        ),
        optax.add_decayed_weights(
            weight_decay=weight_decay,
        ),
    ]

    if clip_grad is not None:
        chain.insert(0, optax.clip_by_global_norm(clip_grad))
    tx = optax.chain(*chain)
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, gradient_accumulation_steps)
    return tx, scheduler_combined


def get_rmsprop_with_warmup_cosine_scheduler(
    steps: int,
    learning_rate: float = 5e-5,
    learning_rate_end: float = 1e-5,
    decay: float = 0.9,
    initial_scale: float = 0.0,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    eps: float = 1e-8,
    weight_decay: float = 1e-1,
    exponent: float = 1.0,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 100,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates an RMSprop optimizer with a cosine learning rate scheduler with warmup.

    Args:
        steps: The total number of training steps.
        learning_rate: The peak learning rate.
        learning_rate_end: The final learning rate.
        decay: The decay rate for the moving average of the squared gradients.
        initial_scale: The initial scale for the moving average of the squared gradients.
        momentum: The momentum value to use. If None, momentum is not used.
        nesterov: Whether to use Nesterov momentum.
        eps: A small value added to the denominator for numerical stability.
        weight_decay: The weight decay rate.
        exponent: The exponent to use for the cosine decay.
        gradient_accumulation_steps: The number of steps to accumulate gradients over.
        warmup_steps: The number of warmup steps.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        A tuple containing the optimizer and the learning rate scheduler.
    """
    for kwarg in kwargs.keys():
        warnings.warn(f"Key {kwarg} is not used for optimizer.")
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.5e-7,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=steps,
        end_value=learning_rate_end,
        exponent=exponent,
    )

    chain = [
        optax.scale_by_rms(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
        (
            optax.trace(
                decay=momentum,
                nesterov=nesterov,
            )
            if momentum is not None
            else optax.identity()
        ),
        optax.add_decayed_weights(
            weight_decay=weight_decay,
        ),
    ]

    if clip_grad is not None:
        chain.insert(0, optax.clip_by_global_norm(clip_grad))
    tx = optax.chain(*chain)
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, gradient_accumulation_steps)
    return tx, scheduler
