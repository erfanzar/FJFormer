import importlib.util
import jax
import flax


def is_torch_available():
    return True if importlib.util.find_spec('torch') is not None else False


def count_num_params(_p):
    return sum(i.size for i in jax.tree_util.tree_flatten(flax.core.unfreeze(_p))[0])


def count_params(_p):
    print('\033[1;31mModel Contain : ', count_num_params(_p) / 1e9, ' Billion Parameters')
