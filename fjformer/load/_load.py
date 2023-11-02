import os
import msgpack
from flax.serialization import from_bytes, to_bytes, to_state_dict
from flax.traverse_util import flatten_dict

from .streamer import StreamingCheckpointer
from jax import numpy as jnp


def get_float_dtype_by_name(dtype):
    return {
        'bf16': jnp.bfloat16,
        'bfloat16': jnp.bfloat16,
        'fp16': jnp.float16,
        'float16': jnp.float16,
        'fp32': jnp.float32,
        'float32': jnp.float32,
        'fp64': jnp.float64,
        'float64': jnp.float64,
    }[dtype]


def float_tensor_to_dtype(tensor, dtype):
    if dtype is None or dtype == '':
        return tensor
    if isinstance(dtype, str):
        dtype = get_float_dtype_by_name(dtype)
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    if getattr(tensor, 'dtype', None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


def load_and_convert_checkpoint_to_torch(path, dtype=jnp.float16, transpose_needed=None,
                                         transpose_not_needed=None, select_params_field: bool = True):
    import torch
    if transpose_needed is None:
        transpose_needed = ["kernel"]
    if transpose_not_needed is None:
        transpose_not_needed = ['none']

    def match_keywords(string, ts, ns):
        for t in ts:
            if t not in string:
                return False
        for n in ns:
            if n in string:
                return False
        return True

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
    flatten_train_state = flatten_dict(train_state)
    if gather_fns is not None:
        gather_fns = flatten_dict(to_state_dict(gather_fns))

    with open(path, "wb") as stream:
        for key, value in flatten_train_state.items():
            if gather_fns is not None:
                value = gather_fns[key](value)
            value = float_tensor_to_dtype(value, float_dtype)
            stream.write(packer.pack((key, to_bytes(value))))
