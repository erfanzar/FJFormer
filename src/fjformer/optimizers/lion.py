from typing import Optional
import chex
import optax


def get_lion_with_linear_scheduler(
        steps: int,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        b1: float = 0.9,
        b2: float = 0.99,
        gradient_accumulation_steps: int = 1,
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
        gradient_accumulation_steps:gradient_accumulation_steps
    Return:
        Optimizer , Scheduler
         """
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
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler


def get_lion_with_with_warmup_linear_scheduler(
        steps: int,
        b1: float = 0.9,
        b2: float = 0.99,
        gradient_accumulation_steps: int = 1,
        mu_dtype: Optional[chex.ArrayDType] = None,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        warmup_steps: int = 500
):
    """

    :param b1:
    :param b2:
    :param mu_dtype:
    :param learning_rate_start:
    :param learning_rate_end:
    :param warmup_steps:
    :param gradient_accumulation_steps:
    :param steps:
    :param gradient_accumulation_steps
    :return: Optimizer and Scheduler
    """
    scheduler_warmup = optax.linear_schedule(init_value=5e-8, end_value=learning_rate_start,
                                             transition_steps=warmup_steps)
    scheduler_decay = optax.linear_schedule(init_value=learning_rate_start, end_value=learning_rate_end,
                                            transition_steps=steps - warmup_steps)

    scheduler_combined = optax.join_schedules(schedules=[scheduler_warmup, scheduler_decay], boundaries=[warmup_steps])
    tx = optax.chain(
        optax.scale_by_lion(
            b1=b1,
            b2=b2,
            mu_dtype=mu_dtype
        ),
        optax.scale_by_schedule(scheduler_combined),
        optax.scale(-1)
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler_combined


def get_lion_with_cosine_scheduler(
        steps: int,
        learning_rate=5e-5,
        alpha: float = 0.0,
        exponent: float = 1.0,
        b1: float = 0.9,
        b2: float = 0.99,
        gradient_accumulation_steps: int = 1,
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
        gradient_accumulation_steps:gradient_accumulation_steps
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
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler


def get_lion_with_warm_up_cosine_scheduler(
        steps: int,
        learning_rate=5e-5,
        exponent: float = 1.0,
        b1: float = 0.9,
        b2: float = 0.99,
        gradient_accumulation_steps: int = 1,
        mu_dtype: Optional[chex.ArrayDType] = None,
):
    """

    :param steps:
    :param learning_rate:
    :param exponent:
    :param b1:
    :param b2:
    :param gradient_accumulation_steps:
    :param mu_dtype:
    :return:
    """
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.5e-7,
        peak_value=learning_rate,
        warmup_steps=steps,
        decay_steps=steps + 1,
        end_value=learning_rate,
        exponent=exponent,
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
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler
