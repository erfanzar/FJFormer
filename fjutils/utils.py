import os

import jax
import msgpack
from jax import numpy as jnp
import numpy as np
import re
from jax.experimental.pjit import PartitionSpec as PS
import flax
from jax.interpreters import pxla
from fjutils.easylm import with_sharding_constraint
from flax.serialization import from_bytes, to_bytes, to_state_dict, from_state_dict
from flax.traverse_util import flatten_dict, unflatten_dict
from fjutils.easylm import float_tensor_to_dtype


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


def count_num_params(_p):
    return sum(i.size for i in jax.tree_util.tree_flatten(flax.core.unfreeze(_p))[0])


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


def read_ckpt(path: [str, os.PathLike], shard_fns=None, add_extra_past_fix: list = None):
    tensors = {}
    with open(path, 'rb') as stream:
        unpacker = msgpack.Unpacker(stream, read_size=83886080, max_buffer_size=0)
        for key, value in unpacker:
            if add_extra_past_fix is not None:
                key = add_extra_past_fix + key
            key = tuple(key)
            tensor = from_bytes(None, value)
            if shard_fns is not None:
                tensor = shard_fns[key](tensor)
            tensors[key] = tensor
    return tensors


def save_ckpt(train_state, path, gather_fns=None, float_dtype=None):
    train_state = to_state_dict(train_state)
    packer = msgpack.Packer()
    flattend_train_state = flatten_dict(train_state)
    if gather_fns is not None:
        gather_fns = flatten_dict(to_state_dict(gather_fns))

    with open(path, "wb") as stream:
        for key, value in flattend_train_state.items():
            if gather_fns is not None:
                value = gather_fns[key](value)
            value = float_tensor_to_dtype(value, float_dtype)
            stream.write(packer.pack((key, to_bytes(value))))
