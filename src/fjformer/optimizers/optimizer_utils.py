from typing import Callable, NamedTuple, Optional

import chex
import jax
import optax
from jax import numpy as jnp


class OptaxScheduledWeightDecayState(NamedTuple):
    """
    State for the scheduled weight decay optimizer.
    """

    count: chex.Array  # Step count


def optax_add_scheduled_weight_decay(
    schedule_fn: Callable[[chex.Array], chex.Array],
    mask: Optional[chex.ArrayTree] = None,
) -> optax.GradientTransformation:
    """
    Create an optax optimizer that applies weight decay on a schedule.

    This function is similar to `optax.add_decayed_weights`, but it allows for
    the weight decay rate to be scheduled over training steps.

    Args:
        schedule_fn: A function that takes the current step count as input
                      and returns the weight decay rate.
        mask: A PyTree with the same structure as the parameters.
              A value of True at a particular location indicates that weight
              decay should be applied to that parameter.

    Returns:
        An `optax.GradientTransformation` object representing the optimizer.
    """

    def init_fn(params: chex.ArrayTree) -> OptaxScheduledWeightDecayState:
        """
        Initializes the state of the optimizer.
        """
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(
        updates: chex.ArrayTree,
        state: OptaxScheduledWeightDecayState,
        params: Optional[chex.ArrayTree] = None,
    ) -> tuple[chex.ArrayTree, OptaxScheduledWeightDecayState]:
        """
        Applies weight decay to the updates based on the schedule.
        """
        if params is None:
            raise ValueError("Params cannot be None for weight decay!")

        weight_decay = schedule_fn(state.count)  # Get scheduled decay rate
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * p, updates, params
        )
        return updates, OptaxScheduledWeightDecayState(
            count=optax.safe_int32_increment(state.count)
        )

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)
