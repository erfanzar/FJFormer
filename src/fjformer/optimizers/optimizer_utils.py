import optax
from typing import NamedTuple
from jax import numpy as jnp
import jax
import chex


class OptaxScheduledWeightDecayState(NamedTuple):
    count: chex.Array


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
