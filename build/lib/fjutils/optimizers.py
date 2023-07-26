import optax
from typing import Optional, NamedTuple, Any
import chex
from jax import numpy as jnp
import jax


class OptaxScheduledWeightDecayState(NamedTuple):
    count: jnp.DeviceArray


def optax_add_scheduled_weight_decay(schedule_fn, mask=None):
    """

    :param schedule_fn:
    :param mask:
    :return: Optax GradientTransformation inited
    """

    def init_fn(params):
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError('Params cannot be None for weight decay!')

        weight_decay = schedule_fn(state.count)
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * p, updates, params
        )
        return updates, OptaxScheduledWeightDecayState(
            count=optax.safe_int32_increment(state.count)
        )

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)


def get_adamw_with_cosine_scheduler(
        steps: int,
        learning_rate: float = 5e-5,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        weight_decay: float = 1e-1,
        mu_dtype: Optional[chex.ArrayDType] = None,

):
    """

    :param steps:
    :param learning_rate:
    :param b1:
    :param b2:
    :param eps:
    :param eps_root:
    :param weight_decay:
    :param mu_dtype:
    :return: Optimizer and Scheduler
    """
    scheduler = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=steps
    )
    tx = optax.chain(
        optax.scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype
        ),
        optax.add_decayed_weights(
            weight_decay=weight_decay
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1)
    )
    return tx, scheduler


def get_adamw_with_linear_scheduler(
        steps: int,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        weight_decay: float = 1e-1,
        mu_dtype: Optional[chex.ArrayDType] = None,

):
    """

    :param steps:
    :param learning_rate_start:
    :param learning_rate_end:
    :param b1:
    :param b2:
    :param eps:
    :param eps_root:
    :param weight_decay:
    :param mu_dtype:
    :return: Optimizer and Scheduler
    """
    scheduler = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps
    )
    tx = optax.chain(
        optax.scale_by_adam(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype
        ),
        optax.add_decayed_weights(
            weight_decay=weight_decay
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1)
    )
    return tx, scheduler


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
        dtype_momentum: Any = jnp.float32,
        weight_decay_rate: Optional[float] = None,
        eps: float = 1e-30,
        factored: bool = True,
        weight_decay_mask=None,

):
    """

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
    return tx, scheduler


def get_adafactor_with_cosine_scheduler(
        steps: int,
        learning_rate=5e-5,
        weight_decay=1e-1,
        min_dim_size_to_factor: int = 128,
        decay_rate: float = 0.8,
        decay_offset: int = 0,
        multiply_by_parameter_scale: float = True,
        clipping_threshold: Optional[float] = 1.0,
        momentum: Optional[float] = None,
        dtype_momentum: Any = jnp.float32,
        weight_decay_rate: Optional[float] = None,
        eps: float = 1e-30,
        factored: bool = True,
        weight_decay_mask=None,

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
    :param weight_decay_mask:
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
        ),
        optax_add_scheduled_weight_decay(
            lambda step: -scheduler(step) * weight_decay,
            weight_decay_mask
        )
    )
    return tx, scheduler


def get_lion_with_cosine_scheduler(
        steps: int,
        learning_rate=5e-5,
        alpha: float = 0.0,
        exponent: float = 1.0,
        b1: float = 0.9,
        b2: float = 0.99,
        mu_dtype: Optional[chex.ArrayDType] = None,
):
    """

   Args:
        learning_rate: An initial value `init_v`.
        steps: Positive integer - the number of steps for which to apply
          the decay for.
        alpha: Float. The minimum value of the multiplier used to adjust the
          learning rate.
        exponent: Float. The default decay is 0.5 * (1 + cos(pi * t/T)), where t is
          the current timestep and T is the `decay_steps`. The exponent modifies
          this to be (0.5 * (1 + cos(pi * t/T))) ** exponent. Defaults to 1.0.
        b1: Rate for combining the momentum and the current grad.
        b2: Decay rate for the exponentially weighted average of grads.
        mu_dtype: Optional `dtype` to be used for the momentum; if
          `None` then the `dtype is inferred from `params` and `updates`.
    Return:
        Optimizer , Scheduler
    """
    try:
        scheduler = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=steps,
            alpha=alpha,
            exponent=exponent
        )
    except:
        scheduler = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=steps,
            alpha=alpha,
            # exponent=exponent
        )
    tx = optax.chain(
        optax.scale_by_lion(
            b1=b1,
            b2=b2,
            mu_dtype=mu_dtype
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1)
    )
    return tx, scheduler


def get_lion_with_linear_scheduler(
        steps: int,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        b1: float = 0.9,
        b2: float = 0.99,
        mu_dtype: Optional[chex.ArrayDType] = None,
):
    """
    Args:
        steps: total train steps (max_steps)
        learning_rate_start: start learning rate for sure
        learning_rate_end: end learning rate for sure :\
        b1: Rate for combining the momentum and the current grad.
        b2: Decay rate for the exponentially weighted average of grads.
        mu_dtype: Optional `dtype` to be used for the momentum; if
          `None` then the `dtype is inferred from `params` and `updates`.
    Return:
        Optimizer , Scheduler"""
    scheduler = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps
    )
    tx = optax.chain(
        optax.scale_by_lion(
            b1=b1,
            b2=b2,
            mu_dtype=mu_dtype
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1)
    )
    return tx, scheduler
