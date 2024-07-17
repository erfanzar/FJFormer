import warnings
from typing import Optional, Tuple

import chex
import optax


def _get_lion_base(
    scheduler: optax.Schedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[chex.ArrayDType] = None,
    gradient_accumulation_steps: int = 1,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> optax.GradientTransformation:
    """
    Creates a base Lion optimizer with the given scheduler.

    Args:
        scheduler (optax.Schedule): Learning rate scheduler.
        b1 (float): The exponential decay rate for the first moment estimates.
        b2 (float): The exponential decay rate for the second moment estimates.
        mu_dtype (Optional[chex.ArrayDType]): Optional datatype for the first moment estimates.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        optax.GradientTransformation: The configured optimizer.
    """
    for kwarg in kwargs.keys():
        warnings.warn(f"Key {kwarg} is not used for optimizer.")
    chain = [
        optax.scale_by_lion(b1=b1, b2=b2, mu_dtype=mu_dtype),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    ]
    if clip_grad is not None:
        chain.insert(0, optax.clip_by_global_norm(clip_grad))
    tx = optax.chain(*chain)
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, gradient_accumulation_steps)
    return tx


def get_lion_with_linear_scheduler(
    steps: int,
    learning_rate_start: float = 5e-5,
    learning_rate_end: float = 1e-5,
    b1: float = 0.9,
    b2: float = 0.99,
    gradient_accumulation_steps: int = 1,
    mu_dtype: Optional[chex.ArrayDType] = None,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates a Lion optimizer with a linear learning rate scheduler.

    Args:
        steps (int): Total number of training steps.
        learning_rate_start (float): Initial learning rate.
        learning_rate_end (float): Final learning rate.
        b1 (float): The exponential decay rate for the first moment estimates.
        b2 (float): The exponential decay rate for the second moment estimates.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        mu_dtype (Optional[chex.ArrayDType]): Optional datatype for the first moment estimates.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        Tuple[optax.GradientTransformation, optax.Schedule]: The optimizer and scheduler.
    """
    scheduler = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps,
    )
    tx = _get_lion_base(
        scheduler=scheduler,
        b1=b1,
        b2=b2,
        mu_dtype=mu_dtype,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        **kwargs,
    )
    return tx, scheduler


def get_lion_with_warmup_linear_scheduler(
    steps: int,
    learning_rate_start: float = 5e-5,
    learning_rate_end: float = 1e-5,
    b1: float = 0.9,
    b2: float = 0.99,
    gradient_accumulation_steps: int = 1,
    mu_dtype: Optional[chex.ArrayDType] = None,
    warmup_steps: int = 100,
    warmup_init_value: float = 5e-8,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates a Lion optimizer with a warm-up linear learning rate scheduler.

    Args:
        steps (int): Total number of training steps.
        learning_rate_start (float): Learning rate after warm-up.
        learning_rate_end (float): Final learning rate.
        b1 (float): The exponential decay rate for the first moment estimates.
        b2 (float): The exponential decay rate for the second moment estimates.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        mu_dtype (Optional[chex.ArrayDType]): Optional datatype for the first moment estimates.
        warmup_steps (int): Number of warm-up steps.
        warmup_init_value (float): Initial learning rate for warm-up.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        Tuple[optax.GradientTransformation, optax.Schedule]: The optimizer and scheduler.
    """
    scheduler_warmup = optax.linear_schedule(
        init_value=warmup_init_value,
        end_value=learning_rate_start,
        transition_steps=warmup_steps,
    )
    scheduler_decay = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps - warmup_steps,
    )

    scheduler_combined = optax.join_schedules(
        schedules=[scheduler_warmup, scheduler_decay], boundaries=[warmup_steps]
    )
    tx = _get_lion_base(
        scheduler=scheduler_combined,
        b1=b1,
        b2=b2,
        mu_dtype=mu_dtype,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        **kwargs,
    )
    return tx, scheduler_combined


def get_lion_with_cosine_scheduler(
    steps: int,
    learning_rate: float = 5e-5,
    alpha: float = 0.0,
    exponent: float = 1.0,
    b1: float = 0.9,
    b2: float = 0.99,
    gradient_accumulation_steps: int = 1,
    mu_dtype: Optional[chex.ArrayDType] = None,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates a Lion optimizer with a cosine learning rate scheduler.

    Args:
        steps (int): Total number of training steps.
        learning_rate (float): Peak learning rate.
        alpha (float): Minimum learning rate as a fraction of learning_rate.
        exponent (float): Exponent for the cosine decay.
        b1 (float): The exponential decay rate for the first moment estimates.
        b2 (float): The exponential decay rate for the second moment estimates.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        mu_dtype (Optional[chex.ArrayDType]): Optional datatype for the first moment estimates.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        Tuple[optax.GradientTransformation, optax.Schedule]: The optimizer and scheduler.
    """
    try:
        scheduler = optax.cosine_decay_schedule(
            init_value=learning_rate, decay_steps=steps, alpha=alpha, exponent=exponent
        )
    except TypeError:
        scheduler = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=steps,
            alpha=alpha,
        )
    tx = _get_lion_base(
        scheduler=scheduler,
        b1=b1,
        b2=b2,
        mu_dtype=mu_dtype,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        **kwargs,
    )
    return tx, scheduler


def get_lion_with_warmup_cosine_scheduler(
    steps: int,
    learning_rate: float = 5e-5,
    learning_rate_end: float = 1e-5,
    exponent: float = 1.0,
    b1: float = 0.9,
    b2: float = 0.99,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 100,
    mu_dtype: Optional[chex.ArrayDType] = None,
    warmup_init_value: float = 0.5e-7,
    clip_grad: Optional[float] = None,
    **kwargs,
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Creates a Lion optimizer with a warm-up cosine learning rate scheduler.

    Args:
        steps (int): Total number of training steps.
        learning_rate (float): Peak learning rate after warm-up.
        learning_rate_end (float): Final learning rate.
        exponent (float): Exponent for the cosine decay.
        b1 (float): The exponential decay rate for the first moment estimates.
        b2 (float): The exponential decay rate for the second moment estimates.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        warmup_steps (int): Number of warm-up steps.
        mu_dtype (Optional[chex.ArrayDType]): Optional datatype for the first moment estimates.
        warmup_init_value (float): Initial learning rate for warm-up.
        clip_grad (Optional[float]): If provided, gradients will be clipped to this maximum norm.

    Returns:
        Tuple[optax.GradientTransformation, optax.Schedule]: The optimizer and scheduler.
    """
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=warmup_init_value,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=steps,
        end_value=learning_rate_end,
        exponent=exponent,
    )
    tx = _get_lion_base(
        scheduler=scheduler,
        b1=b1,
        b2=b2,
        mu_dtype=mu_dtype,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        **kwargs,
    )
    return tx, scheduler
