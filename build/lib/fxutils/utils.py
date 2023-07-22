import jax
from jax import numpy as jnp
import numpy as np
import re
from jax.sharding import PartitionSpec as PS
import flax
from jax.interpreters import pxla
from fjutils.easylm import with_sharding_constraint


def match_partition_rules(rules, params):
    def get_partition_spec(name, leaf):
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {name}')

    def tree_path_to_string(path):
        keys = []
        for i, key in enumerate(path):
            if isinstance(key, jax.tree_util.SequenceKey):
                keys.append(str(key.idx))
            elif isinstance(key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)):
                keys.append(str(key.key))
            elif isinstance(key, jax.tree_util.GetAttrKey):
                keys.append(str(key.name))
            else:
                keys.append(str(key))
        return '/'.join(keys)

    return jax.tree_util.tree_map_with_path(
        lambda path, p: get_partition_spec(tree_path_to_string(path), p),
        params
    )


def count_params(_p):
    print('\033[1;31mModel Contain : ',
          sum(i.size for i in jax.tree_util.tree_flatten(flax.core.unfreeze(_p))[0]) / 1e9, ' Billion Parameters')


def names_in_mesh(*names):
    return set(names) <= set(pxla.thread_resources.env.physical_mesh.axis_names)


def get_names(partition_specs):
    names = set()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
    return list(names)


def with_sharding_constraint__a(x, partition_spec):
    names = get_names(partition_spec)
    if names_in_mesh(*names):
        x = with_sharding_constraint(x, partition_spec)
    return x


def get_devices(tensor):
    return tensor.devices()


def change_to_bf16(tensor):
    return tensor.astype(jnp.bfloat16)


def change_to_fp16(tensor):
    return tensor.astype(jnp.float16)


def change_to_fp32(tensor):
    return tensor.astype(jnp.float32)


def change(tensor, device):
    return jax.device_put(tensor, device)
