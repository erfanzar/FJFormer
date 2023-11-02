from typing import Optional
import jax.numpy as jnp
import chex
import optax
from .optimizer_utils import optax_add_scheduled_weight_decay, OptaxScheduledWeightDecayState


def get_adafactor_with_linear_scheduler(
        steps: int,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        weight_decay=1e-1,
        min_dim_size_to_factor: int = 128,
        decay_rate: float = 0.8,
        decay_offset: int = 0,
        multiply_by_parameter_scale: float = True,
        clipping_threshold: Optional[float] = 1.0,
        momentum: Optional[float] = None,
        dtype_momentum: chex.ArrayDType = jnp.float32,
        weight_decay_rate: Optional[float] = None,
        eps: float = 1e-30,
        factored: bool = True,
        gradient_accumulation_steps: int = 1,
        weight_decay_mask=None,

):
    """

    :param gradient_accumulation_steps:
    :param steps:
    :param learning_rate_start:
    :param learning_rate_end:
    :param weight_decay:
    :param min_dim_size_to_factor:
    :param decay_rate:
    :param decay_offset:
    :param multiply_by_parameter_scale:
    :param clipping_threshold:
    :param momentum:
    :param dtype_momentum:
    :param weight_decay_rate:
    :param eps:
    :param factored:
    :param weight_decay_mask:
    :return: Optimizer and Scheduler
    """
    scheduler = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps
    )

    tx = optax.chain(
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
            factored=factored
        ),
        optax_add_scheduled_weight_decay(
            lambda step: -scheduler(step) * weight_decay,
            weight_decay_mask
        )
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler


def get_adafactor_with_warmup_linear_scheduler(
        steps: int,
        min_dim_size_to_factor: int = 128,
        decay_rate: float = 0.8,
        decay_offset: int = 0,
        multiply_by_parameter_scale: float = True,
        clipping_threshold: Optional[float] = 1.0,
        momentum: Optional[float] = None,
        dtype_momentum: chex.ArrayDType = jnp.float32,
        weight_decay_rate: Optional[float] = None,
        eps: float = 1e-30,
        factored: bool = True,
        gradient_accumulation_steps: int = 1,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        warmup_steps: int = 500
):
    """
    :param min_dim_size_to_factor:
    :param decay_rate:
    :param decay_offset:
    :param multiply_by_parameter_scale:
    :param clipping_threshold:
    :param momentum:
    :param dtype_momentum:
    :param weight_decay_rate:
    :param factored:
    :param warmup_steps:
    :param gradient_accumulation_steps:
    :param steps:
    :param learning_rate_start:
    :param learning_rate_end:
    :param eps:
    :param weight_decay:

     # New parameter for warmup
     @warmup_steps (int): Number of steps for the warmup phase

     # return Optimizer and Scheduler with WarmUp feature
   """

    scheduler_warmup = optax.linear_schedule(init_value=5e-8, end_value=learning_rate_start,
                                             transition_steps=warmup_steps)
    scheduler_decay = optax.linear_schedule(init_value=learning_rate_start, end_value=learning_rate_end,
                                            transition_steps=steps - warmup_steps)

    scheduler_combined = optax.join_schedules(schedules=[scheduler_warmup, scheduler_decay], boundaries=[warmup_steps])

    tx = optax.chain(
        optax.adafactor(
            learning_rate=scheduler_combined,
            min_dim_size_to_factor=min_dim_size_to_factor,
            decay_rate=decay_rate,
            decay_offset=decay_offset,
            multiply_by_parameter_scale=multiply_by_parameter_scale,
            clipping_threshold=clipping_threshold,
            eps=eps,
            momentum=momentum,
            weight_decay_rate=weight_decay_rate,
            dtype_momentum=dtype_momentum,
            factored=factored
        )
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler_combined


def get_adafactor_with_cosine_scheduler(
        steps: int,
        learning_rate=5e-5,
        min_dim_size_to_factor: int = 128,
        decay_rate: float = 0.8,
        decay_offset: int = 0,
        multiply_by_parameter_scale: float = True,
        clipping_threshold: Optional[float] = 1.0,
        momentum: Optional[float] = None,
        dtype_momentum: chex.ArrayDType = jnp.float32,
        weight_decay_rate: Optional[float] = None,
        eps: float = 1e-30,
        factored: bool = True,
        gradient_accumulation_steps: int = 1
):
    """

    :param gradient_accumulation_steps:
    :param steps:
    :param learning_rate:
    :param weight_decay:
    :param min_dim_size_to_factor:
    :param decay_rate:
    :param decay_offset:
    :param multiply_by_parameter_scale:
    :param clipping_threshold:
    :param momentum:
    :param dtype_momentum:
    :param weight_decay_rate:
    :param eps:
    :param factored:
    :param weight_decay_mask:
    :param gradient_accumulation_steps
    :return: Optimizer and Scheduler
    """
    scheduler = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=steps
    )
    tx = optax.chain(
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
            factored=factored
        )
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler


def get_adafactor_with_warm_up_cosine_scheduler(
        steps: int,
        learning_rate=5e-5,
        weight_decay=1e-1,
        min_dim_size_to_factor: int = 128,
        decay_rate: float = 0.8,
        decay_offset: int = 0,
        multiply_by_parameter_scale: float = True,
        clipping_threshold: Optional[float] = 1.0,
        momentum: Optional[float] = None,
        dtype_momentum: chex.ArrayDType = jnp.float32,
        weight_decay_rate: Optional[float] = None,
        eps: float = 1e-30,
        factored: bool = True,
        exponent: float = 1.0,
        weight_decay_mask=None,
        gradient_accumulation_steps: int = 1
):
    """

    :param steps:
    :param learning_rate:
    :param weight_decay:
    :param min_dim_size_to_factor:
    :param decay_rate:
    :param decay_offset:
    :param multiply_by_parameter_scale:
    :param clipping_threshold:
    :param momentum:
    :param dtype_momentum:
    :param weight_decay_rate:
    :param eps:
    :param factored:
    :param exponent:
    :param weight_decay_mask:
    :param gradient_accumulation_steps:
    :return:
    """
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.5e-7,
        peak_value=learning_rate,
        warmup_steps=steps,
        decay_steps=steps + 1,
        end_value=learning_rate,
        exponent=exponent
    )
    tx = optax.chain(
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
            factored=factored
        ),
        optax_add_scheduled_weight_decay(
            lambda step: -scheduler(step) * weight_decay,
            weight_decay_mask
        )
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler
