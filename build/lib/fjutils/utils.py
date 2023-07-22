import importlib.util
import os

import jax
import msgpack
from typing import List, Optional, Callable, Any

from fjutils.checkpointing import StreamingCheckpointer
from jax import numpy as jnp
import numpy as np
import json
import re
from jax.sharding import PartitionSpec as PS
import flax
from jax.interpreters import pxla
from fjutils.easylm import with_sharding_constraint
from flax.serialization import from_bytes, to_bytes, to_state_dict
from flax.traverse_util import flatten_dict
from fjutils.easylm import float_tensor_to_dtype


def is_torch_available():
    return True if importlib.util.find_spec('torch') is not None else False


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


def match_keywords(string, ts, ns):
    for t in ts:
        if t not in string:
            return False
    for n in ns:
        if n in string:
            return False
    return True


def load_and_convert_checkpoint(path, dtype=jnp.float16, transpose_needed: List[str] = ["kernel"],
                                transpose_not_needed: List[str] = ['none'], select_params_field: bool = True):
    import torch
    _, flax_params = StreamingCheckpointer.load_trainstate_checkpoint('params::' + path)
    flax_params = flatten_dict(flax_params['params'], sep='.') if select_params_field else flatten_dict(flax_params,
                                                                                                        sep='.')
    torch_params = {}
    for key, tensor in flax_params.items():
        if match_keywords(key, transpose_needed, transpose_not_needed):
            tensor = tensor.T
        tensor = float_tensor_to_dtype(tensor, dtype)
        torch_params[key] = torch.from_numpy(tensor)
    return torch_params


def read_json(path):
    with open(path, "r") as stream:
        return json.load(stream)


def write_json(text, path):
    with open(path, "w") as stream:
        json.dump(text, stream)


def get_dataloader(dataset_or_huggingface_dataset_hub_id: Any, batch_size: int, num_epochs: int,
                   select_hf_dataset_field='train',
                   max_steps: int = None, max_length: int = 4096, dataset_hf_kwargs: dict = {},
                   collate_fn: Callable = None, shuffle: Optional[bool] = None,
                   sampler=None,
                   batch_sampler=None,
                   num_workers: int = 0,
                   pin_memory: bool = False, drop_last: bool = False,
                   timeout: float = 0, worker_init_fn=None,
                   multiprocessing_context=None, generator=None,
                   *, prefetch_factor: Optional[int] = None,
                   persistent_workers: bool = False,
                   pin_memory_device: str = ""):
    if collate_fn is None:
        def collate_fn(batch):
            rs = {}
            for key in batch[0].keys():
                ssp = [jnp.array(f[key])[..., -max_length:] for f in batch]
                rs[key] = jnp.stack(ssp).reshape(-1, ssp[0].shape[-1])
            return rs
    from torch.utils.data import DataLoader
    if isinstance(dataset_or_huggingface_dataset_hub_id, str):
        from datasets import load_dataset
        dataset = load_dataset(dataset_or_huggingface_dataset_hub_id, **dataset_hf_kwargs)[select_hf_dataset_field]
    else:
        dataset = dataset_or_huggingface_dataset_hub_id

    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        shuffle=shuffle,
        timeout=timeout,
        sampler=sampler, batch_sampler=batch_sampler,
        drop_last=drop_last,
        generator=generator, persistent_workers=persistent_workers,
        pin_memory_device=pin_memory_device,
        multiprocessing_context=multiprocessing_context, worker_init_fn=worker_init_fn

    )
    max_steps = num_epochs * len(dataloader) if max_steps is None else max_steps
    return dataloader, max_steps
