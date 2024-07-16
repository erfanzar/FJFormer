import os
from typing import Callable, Literal, Union, Optional, Tuple

import jax
import flax
import msgpack
import safetensors.flax
import tqdm
from flax.serialization import from_bytes, to_bytes, to_state_dict, from_state_dict
from flax.traverse_util import flatten_dict, unflatten_dict, empty_node
from jax import numpy as jnp

from flax import struct


def load_file(filename: Union[str, os.PathLike]) -> Tuple[dict, dict]:
    """
    Load a checkpoint file from the given filename.

    Args:
        filename: The path to the checkpoint file.

    Returns:
        A tuple containing the state dictionary and metadata.
    """
    result = {}
    with safetensors.safe_open(filename, framework="flax") as f:
        metadata = f.metadata()
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result, metadata


def is_flatten(pytree: Union[dict, struct.PyTreeNode]) -> bool:
    """
    Check if the given PyTree is flattened.

    Args:
        pytree: The PyTree to check.

    Returns:
        True if the PyTree is flattened, False otherwise.
    """
    return True if isinstance([k for k in pytree.keys()][0], tuple) else False


def get_dtype(tensor: jax.Array, dtype: Optional[Union[str, jnp.dtype]]) -> jax.Array:
    """
    Get the tensor with the specified data type.

    Args:
        tensor: The input tensor.
        dtype: The desired data type.

    Returns:
        The tensor with the specified data type.
    """
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
    A class to manage saving and loading checkpoints.

    Args:
        checkpoint_dir: The directory to save checkpoints to.
        enable: Whether to enable saving and loading checkpoints.
        float_dtype: The floating-point data type to use for saving checkpoints.
        save_optimizer_state: Whether to save the optimizer state in the checkpoint.
        verbose: Whether to print verbose output.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, os.PathLike],
        enable: bool = True,
        float_dtype: Union[str, jnp.dtype] = "bf16",
        save_optimizer_state: bool = True,
        verbose: bool = False,
    ):
        self.float_dtype = float_dtype
        self.save_optimizer_state = save_optimizer_state
        self.checkpoint_dir = checkpoint_dir
        self.enable = enable
        self.verbose = verbose

    def save_checkpoint(
        self,
        state: struct.PyTreeNode,
        filename: Union[str, os.PathLike],
        gather_fns: Optional[dict[Callable]] = None,
        mismatch_allowed: bool = True,
    ):
        """
        Save a checkpoint to the given filename.

        Args:
            state: The state dictionary to save.
            filename: The filename to save the checkpoint to.
            gather_fns: A dictionary of functions to gather the state before saving.
            mismatch_allowed: Whether to allow mismatches between the state dictionary and gather functions.
        """
        if self.enable:
            path = os.path.join(self.checkpoint_dir, filename)
        else:
            path = "/dev/null"
        self.save_state_to_file(
            state, path, gather_fns, self.float_dtype, mismatch_allowed=mismatch_allowed
        )

    @staticmethod
    def load_checkpoint_safe(
        path: Union[str, os.PathLike],
        target: Optional[struct.PyTreeNode] = None,
        shard_fns: Optional[dict[Callable]] = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
    ) -> Tuple[Union[struct.PyTreeNode, dict], dict]:
        """
        Load a checkpoint from the given path.

        Args:
            path: The path to the checkpoint file.
            target: The target PyTree to load the checkpoint into.
            shard_fns: A dictionary of functions to shard the state after loading.
            verbose: Whether to print verbose output.
            mismatch_allowed: Whether to allow mismatches between the state dictionary and shard functions.

        Returns:
            A tuple containing the loaded state dictionary and metadata.
        """
        shard_functions_mismatch = 0
        state, metadata = load_file(path)
        state = flax.traverse_util.unflatten_dict(state, sep=".")
        state = flax.traverse_util.flatten_dict(state)

        if shard_fns is not None:
            # Example:
            # shard_fns = {"params": {"Dense_0": jax.pmap(lambda x: x[0])}}

            pbar_sharding = tqdm.tqdm(
                list(state.keys()), desc="Sharding State", disable=not verbose
            )
            if not is_flatten(shard_fns):
                shard_fns = flatten_dict(shard_fns)
            for key in list(state.keys()):
                try:
                    callable_func = shard_fns[key]
                    if callable_func is None and not mismatch_allowed:
                        raise KeyError(
                            f"Shard Function {key} is None and NoneType OBJ is not callable."
                        )

                    if callable_func is None:
                        shard_functions_mismatch += 1
                    else:
                        state[key] = callable_func(state[key])
                except KeyError as k_err:
                    if mismatch_allowed:
                        shard_functions_mismatch += 1
                    else:
                        raise KeyError(k_err)
                pbar_sharding.set_postfix(sharding_mismatch=shard_functions_mismatch)
                pbar_sharding.update(1)
        if target is not None:  # noqa
            flattened_target = flatten_dict(
                to_state_dict(target), keep_empty_nodes=True
            )
            for key, value in flattened_target.items():
                if key not in state and value == empty_node:
                    state[key] = value

        state = unflatten_dict(state)
        if target is None:
            return state, metadata

        return from_state_dict(target, state), metadata

    @staticmethod
    def save_checkpoint_safe(
        state: struct.PyTreeNode,
        path: Union[str, os.PathLike],
        gather_fns: Optional[dict[Callable]] = None,
        float_dtype: Optional[Union[str, jnp.dtype]] = None,
        verbose: bool = True,
        mismatch_allowed: bool = True,
        metadata: Optional[dict[str, str]] = None,
    ):
        """
        Save a checkpoint to the given path using SafeTensors.

        Args:
            state: The state dictionary to save.
            path: The path to the checkpoint file.
            gather_fns: A dictionary of functions to gather the state before saving.
            float_dtype: The floating-point data type to use for saving the checkpoint.
            verbose: Whether to print verbose output.
            mismatch_allowed: Whether to allow mismatches between the state dictionary and gather functions.
            metadata: Additional metadata to store in the checkpoint.
        """
        state = to_state_dict(state)
        gather_functions_mismatch = 0
        if is_flatten(state):
            state = unflatten_dict(state)

        if gather_fns is not None:
            # Example:
            # gather_fns = {"params": {"Dense_0": lambda x: x[0]}}

            if not is_flatten(gather_fns):
                gather_fns = flatten_dict(gather_fns)
            state = flatten_dict(state)
            pbar_gather = tqdm.tqdm(
                list(state.keys()), desc="Gathering State", disable=not verbose
            )
            for key in pbar_gather:
                try:
                    callable_func = gather_fns[key]
                    if callable_func is None and not mismatch_allowed:
                        raise KeyError(
                            f"Gather Function {key} is None and NoneType OBJ is not callable."
                        )
                    if callable_func is None:
                        gather_functions_mismatch += 1
                    else:
                        state[key] = callable_func(state[key])
                except KeyError as e:
                    if mismatch_allowed:
                        pbar_gather.set_postfix(
                            gather_mismatch=gather_functions_mismatch
                        )
                    else:
                        raise KeyError(e)
                pbar_gather.update(1)
        state = flax.traverse_util.flatten_dict(state, sep=".")
        for key in list(state.keys()):
            if not isinstance(state[key], jax.Array):
                state[key] = jnp.array(state[key])
            state[key] = get_dtype(state[key], float_dtype)

        safetensors.flax.save_file(tensors=state, filename=path, metadata=metadata)

    @staticmethod
    def save_state_to_file(
        state: struct.PyTreeNode,
        path: Union[str, os.PathLike],
        gather_fns: Optional[dict[Callable]] = None,
        float_dtype: Optional[Union[str, jnp.dtype]] = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
    ):
        """
        Save the state dictionary to a file.

        Args:
            state: The state dictionary to save.
            path: The path to the file to save the state dictionary to.
            gather_fns: A dictionary of functions to gather the state before saving.
            float_dtype: The floating-point data type to use for saving the state dictionary.
            verbose: Whether to print verbose output.
            mismatch_allowed: Whether to allow mismatches between the state dictionary and gather functions.
        """
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
                    try:
                        callable_func = gather_fns[key]
                        if callable_func is None and not mismatch_allowed:
                            raise KeyError(
                                f"Gather Function {key} is None and NoneType OBJ is not callable."
                            )
                        value = (
                            callable_func(value) if callable_func is not None else value
                        )
                        if callable_func is None:
                            gather_functions_mismatch += 1
                    except KeyError as k_err:
                        if mismatch_allowed:
                            gather_functions_mismatch += 1
                        else:
                            raise KeyError(k_err)
                pbar.set_postfix(gather_functions_mismatch=gather_functions_mismatch)
                value = get_dtype(value, float_dtype)
                stream.write(packer.pack((key, to_bytes(value))))

    def save_pickle(self, obj: object, filename: Union[str, os.PathLike]):
        """
        Save an object to a pickle file.

        Args:
            obj: The object to save.
            filename: The filename to save the object to.
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
        gather_fns: dict[Callable],
        metadata: Optional[dict] = None,
        dataset: Optional[object] = None,
        milestone: bool = False,
    ):
        """
        Save all components of a checkpoint.

        Args:
            state: The state dictionary to save.
            gather_fns: A dictionary of functions to gather the state before saving.
            metadata: Additional metadata to save.
            dataset: The dataset to save.
            milestone: Whether to save a milestone checkpoint.
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
        path: Union[str, os.PathLike],
        target: Optional[struct.PyTreeNode] = None,
        shard_fns: Optional[dict[Callable]] = None,
        remove_dict_prefix: Optional[Tuple[str, ...]] = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
    ) -> Union[struct.PyTreeNode, dict]:
        """
        Load a checkpoint from the given path.

        Args:
            path: The path to the checkpoint file.
            target: The target PyTree to load the checkpoint into.
            shard_fns: A dictionary of functions to shard the state after loading.
            remove_dict_prefix: A tuple of strings representing the prefix to remove from the state dictionary keys.
            verbose: Whether to print verbose output.
            mismatch_allowed: Whether to allow mismatches between the state dictionary and shard functions.

        Returns:
            The loaded state dictionary.
        """
        if shard_fns is not None:
            shard_fns = flatten_dict(to_state_dict(shard_fns))
        if remove_dict_prefix is not None:
            remove_dict_prefix = tuple(remove_dict_prefix)
        flatten_state = {}

        shard_functions_mismatch = 0
        with open(path, "rb") as fin:
            unpacker = msgpack.Unpacker(fin, read_size=83886080, max_buffer_size=0)
            pbar = tqdm.tqdm(
                unpacker, disable=not verbose, desc="Loading Checkpoints From File"
            )
            for key, value in pbar:
                key = tuple(key)
                if remove_dict_prefix is not None:
                    if key[: len(remove_dict_prefix)] == remove_dict_prefix:
                        key = key[len(remove_dict_prefix) :]
                    else:
                        continue

                tensor = from_bytes(None, value)
                if shard_fns is not None:
                    try:
                        callable_func = shard_fns[key]
                        if callable_func is None and not mismatch_allowed:
                            raise KeyError(
                                f"Shard Function {key} is None and NoneType OBJ is not callable."
                            )
                        tensor = (
                            callable_func(tensor)
                            if callable_func is not None
                            else tensor
                        )
                        if callable_func is None:
                            shard_functions_mismatch += 1
                    except KeyError as k_err:
                        if mismatch_allowed:
                            shard_functions_mismatch += 1
                        else:
                            raise KeyError(k_err)
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
        path: Union[str, os.PathLike],
        target: Optional[struct.PyTreeNode] = None,
        shard_fns: Optional[dict[Callable]] = None,
    ) -> Union[struct.PyTreeNode, dict]:
        """
        Load a standard flax checkpoint that's not saved with the
        msgpack streaming format.

        Args:
            path: The path to the checkpoint file.
            target: The target PyTree to load the checkpoint into.
            shard_fns: A dictionary of functions to shard the state after loading.

        Returns:
            The loaded state dictionary.
        """
        with open(path, "rb") as fin:
            encoded_bytes = fin.read()

        state_dict = flax.serialization.msgpack_restore(encoded_bytes)
        if shard_fns is not None:
            shard_fns = to_state_dict(shard_fns)
            state_dict = jax.tree_util.tree_map(
                lambda fn, x: fn(x), shard_fns, state_dict
            )

        if target is None:
            return state_dict
        return from_state_dict(target, state_dict)

    @classmethod
    def load_state_checkpoint(
        cls,
        load_type: Literal["state", "state_params", "params", "flax_params"],
        load_path: Union[str, os.PathLike],
        state_target: Optional[struct.PyTreeNode] = None,
        state_shard_fns: Optional[dict[Callable]] = None,
        disallow_state: bool = False,
        mismatch_allowed: bool = True,
    ) -> Tuple[Optional[struct.PyTreeNode], Optional[flax.core.frozen_dict.FrozenDict]]:
        """
        Load a state checkpoint from the given path.

        Args:
            load_type: The type of checkpoint to load.
            load_path: The path to the checkpoint file.
            state_target: The target PyTree to load the state into.
            state_shard_fns: A dictionary of functions to shard the state after loading.
            disallow_state: Whether to disallow loading the full state.
            mismatch_allowed: Whether to allow mismatches between the state dictionary and shard functions.

        Returns:
            A tuple containing the loaded state and parameters.
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
                mismatch_allowed=mismatch_allowed,
            )
        elif load_type == "state_params":
            restored_params = cls.load_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,
                remove_dict_prefix=("params", "params"),
                mismatch_allowed=mismatch_allowed,
            )
            restored_params = flax.core.frozen_dict.freeze({"params": restored_params})
        elif load_type == "params":
            restored_params = cls.load_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,
                mismatch_allowed=mismatch_allowed,
            )
            restored_params = flax.core.frozen_dict.freeze({"params": restored_params})
        elif load_type == "flax_params":
            restored_params = cls.load_flax_checkpoint(
                path=load_path,
                target=params_target,
                shard_fns=params_shard_fns,
            )
            restored_params = flax.core.frozen_dict.freeze({"params": restored_params})
        else:
            raise ValueError(f"Invalid load_from type: {load_type}")

        return state, restored_params
