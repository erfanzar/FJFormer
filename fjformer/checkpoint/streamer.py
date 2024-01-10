import os
import jax
import flax
import tqdm
from flax.serialization import (
    from_bytes, to_bytes, to_state_dict, from_state_dict
)
from flax.traverse_util import flatten_dict, unflatten_dict, empty_node
import msgpack
from jax import numpy as jnp

from flax import struct
from typing import Callable, Literal


def get_dtype(tensor, dtype):
    if dtype is None or dtype == "":
        return tensor
    if isinstance(dtype, str):
        dtype = {
            "bf16": jnp.bfloat16,
            "bfloat16": jnp.bfloat16,
            "fp16": jnp.float16,
            "float16": jnp.float16,
            "fp32": jnp.float32,
            "float32": jnp.float32,
            "fp64": jnp.float64,
            "float64": jnp.float64,
        }[dtype]
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    if getattr(tensor, "dtype", None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


class CheckpointManager(object):
    """
    Custom msgpack checkpointer that saves large train states by serializing
    and saving tensors one by one in a streaming fashion. Avoids running
    out of memory or local TPU disk with default flax checkpointer.
    """

    def __init__(
            self,
            checkpoint_dir,
            enable=True,
            float_dtype: str | jnp.dtype = "bf16",
            save_optimizer_state: bool = True,
            verbose: bool = False
    ):
        self.float_dtype = float_dtype
        self.save_optimizer_state = save_optimizer_state
        self.checkpoint_dir = checkpoint_dir
        self.enable = enable
        self.verbose = verbose

    def save_checkpoint(
            self,
            state: struct.PyTreeNode,
            filename: str | os.PathLike,
            gather_fns: dict[Callable] = None,
            mismatch_allowed: bool = True

    ):
        if self.enable:
            path = os.path.join(self.checkpoint_dir, filename)
        else:
            path = "/dev/null"
        self.save_state_to_file(
            state, path, gather_fns, self.float_dtype, mismatch_allowed=mismatch_allowed
        )

    @staticmethod
    def save_state_to_file(
            state: struct.PyTreeNode,
            path: str | os.PathLike,
            gather_fns: dict[Callable] = None,
            float_dtype=None,
            verbose: bool = False,
            mismatch_allowed: bool = True
    ):
        state = to_state_dict(state)
        packer = msgpack.Packer()
        flatten_state = flatten_dict(state)
        if gather_fns is not None:
            gather_fns = flatten_dict(to_state_dict(gather_fns))
        pbar = tqdm.tqdm(
            flatten_state.items(),
            disable=not verbose,
            desc="Saving State to File",
        )

        gather_functions_mismatch = 0

        with open(path, "wb") as stream:
            for key, value in pbar:
                if gather_fns is not None:
                    callable_func = gather_fns[key]
                    if callable_func is None and not mismatch_allowed:
                        raise KeyError(f"Gather Function {key} is None and NoneType OBJ is not callable.")
                    value = callable_func(value) if callable_func is not None else value
                    if callable_func is None:
                        gather_functions_mismatch += 1
                    pbar.set_postfix(gather_functions_mismatch=gather_functions_mismatch)
                value = get_dtype(value, float_dtype)
                stream.write(packer.pack((key, to_bytes(value))))

    def save_pickle(
            self,
            obj,
            filename: str | os.PathLike
    ):
        """
        The save_pickle function saves a Python object to disk using the pickle module.

        :param self: Represent the instance of the class
        :param obj: Pass the object that is to be pickled
        :param filename: Specify the name of the file to be saved
        :return: A pickle object
        
        """
        import pickle

        def save_pickle(obj_, path_):
            with open(path_, "wb") as stream:
                pickle.dump(obj_, stream)

        if self.enable:
            path = os.path.join(self.checkpoint_dir, filename)
        else:
            path = "/dev/null"
        save_pickle(obj, path)

    def save_all(
            self,
            state: struct.PyTreeNode,
            gather_fns,
            metadata=None,
            dataset=None,
            milestone=False
    ):
        """
        The save_all function saves the following:
            - metadata.pkl (a pickle file containing a dictionary of metadata)
            - dataset.pkl (a pickle file containing the training data)
            - streaming_params_{step}.pkl or streaming_state_{step}.pkl
                (depending on whether we want to save optimizer state or not,
                this is a checkpoint that will not be overwritten by future checkpoints)

        :param self: Access the attributes and methods of the class
        :param state: struct.PyTreeNode: Save the current state of the model
        :param gather_fns: Gather the state of the optimizer
        :param metadata: Save the metadata of the training
        :param dataset: Save the dataset to disk
        :param milestone: Determine whether the checkpoint is a milestone or not
        :return: Nothing
        
        """
        step = int(jax.device_get(state.step))
        if self.save_optimizer_state:
            checkpoint_state = state
            checkpoint_name = "streaming_state"
            checkpoint_gather_fns = gather_fns
        else:
            checkpoint_state = state.params["params"]
            checkpoint_name = "streaming_params"
            checkpoint_gather_fns = gather_fns.params["params"]

        if milestone:
            # Save a milestone checkpoint that will not be overwritten
            self.save_pickle(metadata, f"metadata_{step}.pkl")
            self.save_pickle(dataset, f"dataset_{step}.pkl")
            self.save_checkpoint(
                checkpoint_state, f"{checkpoint_name}_{step}", checkpoint_gather_fns
            )
        else:
            # Save a normal checkpoint that can be overwritten
            self.save_pickle(metadata, "metadata.pkl")
            self.save_pickle(dataset, "dataset.pkl")
            self.save_checkpoint(
                checkpoint_state, f"{checkpoint_name}", checkpoint_gather_fns
            )

    @staticmethod
    def load_checkpoint(
            path: str | os.PathLike,
            target=None,
            shard_fns: dict[Callable] = None,
            remove_dict_prefix=None,
            verbose: bool = False,
            mismatch_allowed: bool = True,
    ):
        """
        The load_checkpoint function is used to checkpoint a checkpoint from disk.

        :param path: Specify the path to the checkpoint file
        :param target: Specify the model to checkpoint the checkpoint into
        :param shard_fns: Specify a function that will be applied to each tensor in the checkpoint
        :param remove_dict_prefix: Remove the prefix of a dictionary     
        :param verbose: print state and other stuff
        :param mismatch_allowed: when ever to allow shard_fns to be passed even if their None
        :return:  of the form {key: value}, where key is a tuple and value is a tensor
        
        """
        if shard_fns is not None:
            shard_fns = flatten_dict(
                to_state_dict(shard_fns)
            )
        if remove_dict_prefix is not None:
            remove_dict_prefix = tuple(remove_dict_prefix)
        flatten_state = {}

        shard_functions_mismatch = 0
        with open(path, "rb") as fin:
            unpacker = msgpack.Unpacker(fin, read_size=83886080, max_buffer_size=0)
            pbar = tqdm.tqdm(
                unpacker,
                disable=not verbose,
                desc="Loading Checkpoints From File"
            )
            for key, value in pbar:
                key = tuple(key)
                if remove_dict_prefix is not None:
                    if key[:len(remove_dict_prefix)] == remove_dict_prefix:
                        key = key[len(remove_dict_prefix):]
                    else:
                        continue

                tensor = from_bytes(None, value)
                if shard_fns is not None:
                    callable_func = shard_fns[key]
                    if callable_func is None and not mismatch_allowed:
                        raise KeyError(f"Shard Function {key} is None and NoneType OBJ is not callable.")
                    tensor = callable_func(tensor) if callable_func is not None else tensor
                    if callable_func is None:
                        shard_functions_mismatch += 1
                flatten_state[key] = tensor
                pbar.set_postfix(shard_functions_mismatch=shard_functions_mismatch)
        if target is not None:
            flattened_target = flatten_dict(
                to_state_dict(target), keep_empty_nodes=True
            )
            for key, value in flattened_target.items():
                if key not in flatten_state and value == empty_node:
                    flatten_state[key] = value

        state = unflatten_dict(flatten_state)
        if target is None:
            return state

        return from_state_dict(target, state)

    @staticmethod
    def load_flax_checkpoint(
            path,
            target=None,
            shard_fns=None
    ):
        """ Load a standard flax checkpoint that"s not saved with the
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
    def load_state_checkpoint(
            cls,
            load_type: Literal[
                "state",
                "state_params",
                "params",
                "flax_params"
            ],
            load_path: str | os.PathLike,
            state_target=None,
            state_shard_fns=None,
            disallow_state=False,
            mismatch_allowed: bool = True
    ):
        """
        The load_state_checkpoint function is used to checkpoint a checkpoint from disk.

        :param cls: Call the load_checkpoint function
        :param load_type: Specify which part of state to checkpoint
        :param load_path: Specify where to checkpoint the model from
        :param state_target: Specify the target for the train state
        :param state_shard_fns: Specify the sharding function
        :param disallow_state: Prevent loading the entire state
        :param mismatch_allowed: when ever to allow shard func to be None
        :return: A tuple of two objects, the state and restored_params
        
        """
        if state_target is not None:
            params_target = state_target.params["params"]
        else:
            params_target = None

        if state_shard_fns is not None:
            params_shard_fns = state_shard_fns.params["params"]
        else:
            params_shard_fns = None

        if disallow_state:
            assert load_type != "state", "Loading full state is not allowed!"
        state = None
        restored_params = None
        if load_type == "state":
            state = cls.load_checkpoint(
                path=load_path,
                target=state_target,
                shard_fns=state_shard_fns,
                mismatch_allowed=mismatch_allowed
            )
        elif load_type == "state_params":
            restored_params = cls.load_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,
                remove_dict_prefix=("params", "params"),
                mismatch_allowed=mismatch_allowed
            )
            restored_params = flax.core.frozen_dict.freeze(
                {"params": restored_params}
            )
        elif load_type == "params":
            restored_params = cls.load_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,
                mismatch_allowed=mismatch_allowed
            )
            restored_params = flax.core.frozen_dict.freeze(
                {"params": restored_params}
            )
        elif load_type == "flax_params":
            restored_params = cls.load_flax_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,
            )
            restored_params = flax.core.frozen_dict.freeze(
                {"params": restored_params}
            )
        else:
            raise ValueError(f"Invalid load_from type: {load_type}")

        return state, restored_params
