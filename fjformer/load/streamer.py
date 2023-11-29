import os

from ml_collections import ConfigDict
import jax
import flax
from flax.serialization import (
    from_bytes, to_bytes, to_state_dict, from_state_dict
)
from flax.traverse_util import flatten_dict, unflatten_dict, empty_node
import msgpack
from jax import numpy as jnp


def get_dtype(tensor, dtype):
    if dtype is None or dtype == '':
        return tensor
    if isinstance(dtype, str):
        dtype = {
            'bf16': jnp.bfloat16,
            'bfloat16': jnp.bfloat16,
            'fp16': jnp.float16,
            'float16': jnp.float16,
            'fp32': jnp.float32,
            'float32': jnp.float32,
            'fp64': jnp.float64,
            'float64': jnp.float64,
        }[dtype]
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    if getattr(tensor, 'dtype', None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


class StreamingCheckpointer(object):
    """ Custom msgpack checkpointer that saves large train states by serializing
        and saving tensors one by one in a streaming fashion. Avoids running
        out of memory or local TPU disk with default flax checkpointer.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.float_dtype = 'bf16'
        config.save_optimizer_state = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, checkpoint_dir, enable=True):
        self.config = self.get_default_config(config)
        self.checkpoint_dir = checkpoint_dir
        self.enable = enable

    def save_checkpoint(self, train_state, filename, gather_fns=None):
        if self.enable:
            path = os.path.join(self.checkpoint_dir, filename)
        else:
            path = '/dev/null'
        self.save_train_state_to_file(
            train_state, path, gather_fns, self.config.float_dtype
        )

    @staticmethod
    def save_train_state_to_file(train_state, path, gather_fns=None, float_dtype=None):
        train_state = to_state_dict(train_state)
        packer = msgpack.Packer()
        flattend_train_state = flatten_dict(train_state)
        if gather_fns is not None:
            gather_fns = flatten_dict(to_state_dict(gather_fns))

        with open(path, "wb") as fout:
            for key, value in flattend_train_state.items():
                if gather_fns is not None:
                    value = gather_fns[key](value)
                value = get_dtype(value, float_dtype)
                fout.write(packer.pack((key, to_bytes(value))))

    def save_pickle(self, obj, filename):
        """
        The save_pickle function saves a Python object to disk using the pickle module.

        :param self: Represent the instance of the class
        :param obj: Pass the object that is to be pickled
        :param filename: Specify the name of the file to be saved
        :return: A pickle object
        
        """
        import pickle
        def save_pickle(obj, path):
            with open(path, 'wb') as fout:
                pickle.dump(obj, fout)

        if self.enable:
            path = os.path.join(self.checkpoint_dir, filename)
        else:
            path = '/dev/null'
        save_pickle(obj, path)

    def save_all(self, train_state, gather_fns, metadata=None, dataset=None, milestone=False):
        """
        The save_all function saves the following:
            - metadata.pkl (a pickle file containing a dictionary of metadata)
            - dataset.pkl (a pickle file containing the training data)
            - streaming_params_{step}.pkl or streaming_train_state_{step}.pkl
                (depending on whether we want to save optimizer state or not,
                this is a checkpoint that will not be overwritten by future checkpoints)

        :param self: Access the attributes and methods of the class
        :param train_state: Save the current state of the model
        :param gather_fns: Gather the state of the optimizer
        :param metadata: Save the metadata of the training
        :param dataset: Save the dataset to disk
        :param milestone: Determine whether the checkpoint is a milestone or not
        :return: Nothing
        
        """
        step = int(jax.device_get(train_state.step))
        if self.config.save_optimizer_state:
            checkpoint_state = train_state
            checkpoint_name = 'streaming_train_state'
            checkpoint_gather_fns = gather_fns
        else:
            checkpoint_state = train_state.params['params']
            checkpoint_name = 'streaming_params'
            checkpoint_gather_fns = gather_fns.params['params']

        if milestone:
            # Save a milestone checkpoint that will not be overwritten
            self.save_pickle(metadata, f'metadata_{step}.pkl')
            self.save_pickle(dataset, f'dataset_{step}.pkl')
            self.save_checkpoint(
                checkpoint_state, f'{checkpoint_name}_{step}', checkpoint_gather_fns
            )
        else:
            # Save a normal checkpoint that can be overwritten
            self.save_pickle(metadata, 'metadata.pkl')
            self.save_pickle(dataset, 'dataset.pkl')
            self.save_checkpoint(
                checkpoint_state, f'{checkpoint_name}', checkpoint_gather_fns
            )

    @staticmethod
    def load_checkpoint(path, target=None, shard_fns=None, remove_dict_prefix=None):
        """
        The load_checkpoint function is used to load a checkpoint from disk.

        :param path: Specify the path to the checkpoint file
        :param target: Specify the model to load the checkpoint into
        :param shard_fns: Specify a function that will be applied to each tensor in the checkpoint
        :param remove_dict_prefix: Remove the prefix of a dictionary
        :return: A dictionary of the form {key: value}, where key is a tuple and value is a tensor
        
        """
        if shard_fns is not None:
            shard_fns = flatten_dict(
                to_state_dict(shard_fns)
            )
        if remove_dict_prefix is not None:
            remove_dict_prefix = tuple(remove_dict_prefix)
        flattend_train_state = {}
        with open(path, 'rb') as fin:
            # 83886080 bytes = 80 MB, which is 16 blocks on GCS
            unpacker = msgpack.Unpacker(fin, read_size=83886080, max_buffer_size=0)
            for key, value in unpacker:
                key = tuple(key)
                if remove_dict_prefix is not None:
                    if key[:len(remove_dict_prefix)] == remove_dict_prefix:
                        key = key[len(remove_dict_prefix):]
                    else:
                        continue

                tensor = from_bytes(None, value)
                if shard_fns is not None:
                    tensor = shard_fns[key](tensor)
                flattend_train_state[key] = tensor

        if target is not None:
            flattened_target = flatten_dict(
                to_state_dict(target), keep_empty_nodes=True
            )
            for key, value in flattened_target.items():
                if key not in flattend_train_state and value == empty_node:
                    flattend_train_state[key] = value

        train_state = unflatten_dict(flattend_train_state)
        if target is None:
            return train_state

        return from_state_dict(target, train_state)

    @staticmethod
    def load_flax_checkpoint(path, target=None, shard_fns=None):
        """ Load a standard flax checkpoint that's not saved with the
            msgpack streaming format.
        """
        with open(path, "rb") as fin:
            encoded_bytes = fin.read()

        state_dict = flax.serialization.msgpack_restore(encoded_bytes)
        if shard_fns is not None:
            shard_fns = to_state_dict(shard_fns)
            state_dict = jax.tree_util.tree_map(lambda fn, x: fn(x), shard_fns, state_dict)

        if target is None:
            return state_dict
        return from_state_dict(target, state_dict)

    @classmethod
    def load_trainstate_checkpoint(cls, load_from, trainstate_target=None,
                                   trainstate_shard_fns=None,
                                   disallow_trainstate=False):
        """
        The load_trainstate_checkpoint function is used to load a checkpoint from disk.

        :param cls: Call the load_checkpoint function
        :param load_from: Specify where to load the model from
        :param trainstate_target: Specify the target for the train state
        :param trainstate_shard_fns: Specify the sharding function
        :param disallow_trainstate: Prevent loading the entire trainstate
        :return: A tuple of two objects, the train_state and restored_params
        
        """
        if trainstate_target is not None:
            params_target = trainstate_target.params['params']
        else:
            params_target = None

        if trainstate_shard_fns is not None:
            params_shard_fns = trainstate_shard_fns.params['params']
        else:
            params_shard_fns = None

        load_type, load_path = load_from.split('::', 1)
        if disallow_trainstate:
            assert load_type != 'trainstate', 'Loading full trainstate is not allowed!'
        train_state = None
        restored_params = None
        if load_type == 'trainstate':
            # Load the entire train state in the streaming format
            train_state = cls.load_checkpoint(
                path=load_path,
                target=trainstate_target,
                shard_fns=trainstate_shard_fns,
            )
        elif load_type == 'trainstate_params':
            # Load the params part of the train state in the streaming format
            restored_params = cls.load_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,
                remove_dict_prefix=('params', 'params'),
            )
            restored_params = flax.core.frozen_dict.freeze(
                {'params': restored_params}
            )
        elif load_type == 'params':
            # Load the params in the streaming format
            restored_params = cls.load_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,
            )
            restored_params = flax.core.frozen_dict.freeze(
                {'params': restored_params}
            )
        elif load_type == 'flax_params':
            # Load the params in the standard flax format (non-streaming)
            # This requires the entire params to fit in memory
            restored_params = cls.load_flax_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns
            )
            restored_params = flax.core.frozen_dict.freeze(
                {'params': restored_params}
            )
        else:
            raise ValueError(f'Invalid load_from type: {load_type}')

        return train_state, restored_params
