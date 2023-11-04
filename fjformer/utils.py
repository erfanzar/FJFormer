import importlib.util
import jax
import flax


def is_torch_available():
    return True if importlib.util.find_spec('torch') is not None else False


def count_num_params(_p):
    return sum(i.size for i in jax.tree_util.tree_flatten(flax.core.unfreeze(_p))[0])


def count_params(_p):
    print('\033[1;31mModel Contain : ', count_num_params(_p) / 1e9, ' Billion Parameters')


class JaxRNG(object):
    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


class GenerateRNG:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)

    def __next__(self):
        while True:
            self.rng, ke = jax.random.split(self.rng, 2)
            return ke
