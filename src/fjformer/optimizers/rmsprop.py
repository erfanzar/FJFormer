from typing import Optional
import optax


def get_rmsprop_with_cosine_scheduler(
        steps: int,
        learning_rate: float = 5e-5,
        decay: float = 0.9,
        initial_scale: float = 0.,
        momentum: Optional[float] = None,
        nesterov: bool = False,
        eps: float = 1e-8,
        weight_decay: float = 1e-1,
        gradient_accumulation_steps: int = 1,

):

    """
    The get_rmsprop_with_cosine_scheduler function returns a tuple of the optimizer and scheduler.
    The optimizer is composed of several transformations:
        1) scale_by_rms - scales the gradient by RMS (root-mean-square) values, which are calculated using an
         exponential moving average with decay rate `decay` and initial value `initial_scale`. The epsilon
         parameter prevents division by zero.
        2) scale_by_schedule - scales the gradient by a schedule, in this case cosine decay with initial value
        `learning rate` and number of steps to complete one cycle equal to total number of training steps.


    :param steps: int: Set the number of steps in the cosine decay schedule
    :param learning_rate: float: Set the initial learning rate
    :param decay: float: Control the decay rate of the running average
    :param initial_scale: float: Set the initial scale of the rmsprop optimizer
    :param momentum: Optional[float]: Specify the momentum of the optimizer
    :param nesterov: bool: Determine whether to use the nesterov momentum algorithm
    :param eps: float: Avoid division by zero
    :param weight_decay: float: Add a weight decay to the loss function
    :param gradient_accumulation_steps: int: Accumulate the gradients over multiple steps before updating the weights
    :param : Define the number of steps to be taken before the learning rate is decayed
    :return: The optimizer and the scheduler
    """
    scheduler = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=steps
    )

    tx = optax.chain(
        optax.scale_by_rms(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
        (
            optax.trace(
                decay=momentum,
                nesterov=nesterov
            )
            if momentum is not None else optax.identity()
        ),
        optax.add_decayed_weights(
            weight_decay=weight_decay
        ),
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler


def get_rmsprop_with_linear_scheduler(
        steps: int,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        decay: float = 0.9,
        initial_scale: float = 0.,
        momentum: Optional[float] = None,
        nesterov: bool = False,
        eps: float = 1e-8,
        weight_decay: float = 1e-1,
        gradient_accumulation_steps: int = 1,

):

    """
    The get_rmsprop_with_linear_scheduler function returns a tuple of two objects:
        1. A transformation (tx) that is applied to the gradients before they are used to update the model parameters.
        2. A scheduler object that can be used to retrieve the current learning rate at any point during training.

    :param steps: int: Define how many steps the learning rate will take to transition from learning_rate_start to learning_rate_end
    :param learning_rate_start: float: Set the initial learning rate
    :param learning_rate_end: float: Specify the final learning rate
    :param decay: float: Control the decay rate of the rmsprop algorithm
    :param initial_scale: float: Scale the initial gradient
    :param momentum: Optional[float]: Set the momentum of the optimizer
    :param nesterov: bool: Determine whether to use nesterov momentum or not
    :param eps: float: Prevent division by zero
    :param weight_decay: float: Apply weight decay to the model weights
    :param gradient_accumulation_steps: int: Accumulate the gradients over multiple batches
    :param : Control the learning rate decay
    :return: Optimizer,Scheduler
    """
    scheduler = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps
    )

    tx = optax.chain(
        optax.scale_by_rms(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
        (
            optax.trace(
                decay=momentum,
                nesterov=nesterov
            )
            if momentum is not None else optax.identity()
        ),
        optax.add_decayed_weights(
            weight_decay=weight_decay
        ),
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler


def get_rmsprop_with_warmup_linear_scheduler(
        steps: int,
        learning_rate_start: float = 5e-5,
        learning_rate_end: float = 1e-5,
        decay: float = 0.9,
        eps: float = 1e-8,
        initial_scale: float = 0.,
        momentum: Optional[float] = None,
        nesterov: bool = False,
        weight_decay: float = 1e-1,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 500
):

    """
    The get_rmsprop_with_warmup_linear_scheduler function returns a tuple of the following:
        1. A JAX optimizer transformation (tx) that performs RMSprop with warmup and linear decay, as well as weight decay.
        2. A JAX schedule object (scheduler_combined) that can be used to plot the learning rate over time.

    :param steps: int: Define the number of steps in the training loop
    :param learning_rate_start: float: Set the learning rate at the start of training
    :param learning_rate_end: float: Set the learning rate at the end of training
    :param decay: float: Control the decay rate of the moving average
    :param eps: float: Avoid division by zero
    :param initial_scale: float: Set the initial scale of the rmsprop optimizer
    :param momentum: Optional[float]: Set the momentum of the optimizer
    :param nesterov: bool: Determine whether to use the nesterov momentum algorithm
    :param weight_decay: float: Add a weight decay to the loss function
    :param gradient_accumulation_steps: int: Accumulate the gradients over multiple batches
    :param warmup_steps: int: Set the number of steps to warm up the learning rate
    :return: Optimizer,Scheduler
    """
    scheduler_warmup = optax.linear_schedule(
        init_value=5e-8,
        end_value=learning_rate_start,
        transition_steps=warmup_steps
    )
    scheduler_decay = optax.linear_schedule(
        init_value=learning_rate_start,
        end_value=learning_rate_end,
        transition_steps=steps - warmup_steps
    )

    scheduler_combined = optax.join_schedules(
        schedules=[scheduler_warmup, scheduler_decay],
        boundaries=[warmup_steps]
    )

    tx = optax.chain(
        optax.scale_by_rms(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale
        ),
        optax.scale_by_schedule(scheduler_combined),
        optax.scale(-1),
        (
            optax.trace(
                decay=momentum,
                nesterov=nesterov
            )
            if momentum is not None else optax.identity()
        ),
        optax.add_decayed_weights(
            weight_decay=weight_decay
        ),
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler_combined


def get_rmsprop_with_warm_up_cosine_scheduler(
        steps: int,
        learning_rate: float = 5e-5,
        learning_rate_end: float = 1e-5,
        decay: float = 0.9,
        initial_scale: float = 0.,
        momentum: Optional[float] = None,
        nesterov: bool = False,
        eps: float = 1e-8,
        weight_decay: float = 1e-1,
        exponent: float = 1.0,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 500,
):
    """
    The get_rmsprop_with_warm_up_cosine_scheduler function returns a tuple of two objects:
        1. A transformation (tx) that is applied to the gradients before they are used to update the parameters.
        2. A scheduler object that can be used to get the current learning rate at any given step in training.

    :param steps: int: Define the number of steps in the warm up phase
    :param learning_rate: float: Set the learning rate of the optimizer
    :param learning_rate_end: float: Set the final learning rate of the optimizer after decay
    :param decay: float: Control the decay rate of the rmsprop algorithm
    :param initial_scale: float: Scale the initial gradient
    :param momentum: Optional[float]: Define the momentum of the optimizer
    :param nesterov: bool: Indicate whether to use nesterov momentum
    :param eps: float: Avoid division by zero
    :param weight_decay: float: Add a weight decay to the loss function
    :param exponent: float: Control the shape of the cosine curve
    :param gradient_accumulation_steps: int: Accumulate gradients over multiple batches
    :param warmup_steps: int: Number of steps of the linear warmup
    :param : Control the learning rate
    :return: Optimizer,Scheduler
    """
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.5e-7,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=steps,
        end_value=learning_rate_end,
        exponent=exponent
    )

    tx = optax.chain(
        optax.scale_by_rms(
            decay=decay,
            eps=eps,
            initial_scale=initial_scale
        ),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
        (
            optax.trace(
                decay=momentum,
                nesterov=nesterov
            )
            if momentum is not None else optax.identity()
        ),
        optax.add_decayed_weights(
            weight_decay=weight_decay
        ),
    )
    if gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(
            tx, gradient_accumulation_steps
        )
    return tx, scheduler
