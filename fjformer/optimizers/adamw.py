from typing import Optional
import chex
import optax


def get_adamw_with_cosine_scheduler(
        steps: int,
        learning_rate: float = 5e-5,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        weight_decay: float = 1e-1,
        gradient_accumulation_steps: int = 1,
        mu_dtype: Optional[chex.ArrayDType] = None,

):
    """

    :param gradient_accumulation_steps:
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
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
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
        gradient_accumulation_steps: int = 1,
        mu_dtype: Optional[chex.ArrayDType] = None,

):
    """

    :param gradient_accumulation_steps:
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
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler


def get_adamw_with_warmup_linear_scheduler(
        steps: int,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        weight_decay: float = 1e-1,
        gradient_accumulation_steps: int = 1,
        mu_dtype: Optional[chex.ArrayDType] = None,
        warmup_steps: int = 500
):
    """
    Thanks TO [JinSeoungwoo](https://github.com/erfanzar/EasyDeL/issues/32)
    :param warmup_steps:
    :param gradient_accumulation_steps:
    :param steps:
    :param learning_rate_start:
    :param learning_rate_end:
    :param b1:
    :param b2:
    :param eps:
    :param eps_root:
    :param weight_decay:
    :param mu_dtype:

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
        optax.scale_by_schedule(scheduler_combined),
        optax.scale(-1)
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler_combined


def get_adamw_with_warm_up_cosine_scheduler(
        steps: int,
        learning_rate: float = 5e-5,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        weight_decay: float = 1e-1,
        exponent: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mu_dtype: Optional[chex.ArrayDType] = None
):
    """

    :param steps:
    :param learning_rate:
    :param b1:
    :param b2:
    :param eps:
    :param eps_root:
    :param weight_decay:
    :param exponent:
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
        exponent=exponent
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
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler
