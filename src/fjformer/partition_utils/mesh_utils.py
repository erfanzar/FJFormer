import jax
import jax.numpy as jnp
import re

from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.pjit import pjit, with_sharding_constraint as _with_sharding_constraint
import numpy as np
from jax.sharding import PartitionSpec as PS
from jax.experimental import mesh_utils
from jax.interpreters import pxla
import flax
from jax.sharding import Mesh
from typing import Sequence


def make_shard_and_gather_fns(partition_specs, dtype_specs=None):
    """
    The make_shard_and_gather_fns function takes in a partition_specs and dtype_specs,
    and returns two functions: shard_fns and gather_fns. The shard function is used to
    shard the input tensor into the specified partitions. The gather function is used to
    gather all the shards back together into one tensor.

    :param partition_specs: Specify the sharding of the input tensor
    :param dtype_specs: Specify the dtype of the tensor
    :return: A tuple of functions
    
    """
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

    def make_to_dtype_fn(dtype_spec):
        def to_dtype(tensor):
            if dtype_specs in float_dtypes and getattr(tensor, 'dtype', None) in float_dtypes:
                # force np array to jax numpy array
                return jnp.asarray(tensor).astype(dtype_specs)
            elif hasattr(dtype_spec, 'dtype') and hasattr(tensor, 'dtype'):
                return jnp.asarray(tensor).astype(dtype_spec.dtype)
            return jnp.asarray(tensor)

        return to_dtype

    def make_shard_fn(partition_spec, dtype_spec=None):
        jax_shard_function = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=None,
            out_shardings=partition_spec
        )

        def shard_fn(tensor):
            return jax_shard_function(tensor).block_until_ready()

        return shard_fn

    def make_gather_fn(partition_spec, dtype_spec=None):
        jax_gather_fn = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=partition_spec,
            out_shardings=None
        )

        def gather_fn(tensor):
            return jax.device_get(jax_gather_fn(tensor))

        return gather_fn

    if dtype_specs is None or dtype_specs in float_dtypes:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
    else:
        shard_fns = jax.tree_util.tree_map(
            make_shard_fn, partition_specs, dtype_specs
        )
        gather_fns = jax.tree_util.tree_map(
            make_gather_fn, partition_specs, dtype_specs
        )
    return shard_fns, gather_fns


def get_jax_mesh(axis_dims, names):
    """
    The get_jax_mesh function takes a string of the form:
        &lt;axis_dims&gt;
    where axis_dims is a comma-separated list of dimensions, each dimension being either:
        &lt;name&gt;:&lt;dim&gt;  or  &lt;dim&gt;
    If there are no names, then the default names 'x', 'y', and 'z' will be used. If there are fewer than three dimensions, then the remaining dimensions will be set to 1. For example:

    :param axis_dims: Specify the dimensions of the mesh
    :param names: Specify the names of the dimensions in
    :return: A mesh object
    
    """
    if axis_dims.startswith('!'):
        mesh_axis_splitting = True
        axis_dims = axis_dims[1:]
    else:
        mesh_axis_splitting = False

    if ':' in axis_dims:
        dims = []
        dim_names = []
        for axis in axis_dims.split(','):
            name, dim = axis.split(':')
            assert name in names
            dims.append(int(dim))
            dim_names.append(name)
        assert (set(dim_names) == set(names))
    else:
        dims = [int(x) for x in axis_dims.split(',')]
        dim_names = names
    assert len(dims) == len(names)
    mesh_shape = np.arange(jax.device_count()).reshape(dims).shape
    if mesh_axis_splitting:
        physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
    else:
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(physical_mesh, dim_names)


def names_in_current_mesh(*names):
    """
    The names_in_current_mesh function is used to check if a set of names are in the current mesh.

    :param *names: Pass in a list of names to the function
    :return: A boolean indicating whether
    
    """
    mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
    return set(names) <= set(mesh_axis_names)


def get_names_from_partition_spec(partition_specs):
    """
    The get_names_from_partition_spec function takes a partition_specs argument, which is either a dictionary or list.
    If it's a dictionary, the function converts it to a list of values. Then for each item in the partition_specs list:
        If the item is None, continue (do nothing) and move on to next iteration of loop.
        If the item is an instance of str (i.e., if it's just one string), add that string to names set and move on to next iteration of loop.
        Otherwise (if not None or str), call get_names_from_partition_spec recurs

    :param partition_specs: Specify the partitioning of the data
    :return: A list of names
    
    """
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_partition_spec(item))

    return list(names)


def with_sharding_constraint(x, partition_specs):
    """ A smarter version of with_sharding_constraint that only applies the
        constraint if the current mesh contains the axes in the partition specs.
    """
    axis_names = get_names_from_partition_spec(partition_specs)
    if names_in_current_mesh(*axis_names):
        x = _with_sharding_constraint(x, partition_specs)
    return x


def wrap_function_with_rng(rng):
    """ To be used as decorator, automatically bookkeep a RNG for the wrapped function. """

    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)

        return wrapped

    return wrap_function


def get_metrics(metrics, unreplicate=False, stack=False):
    """
    The get_metrics function is a helper function that takes the metrics dictionary
    returned by the training loop and converts it to a format that can be used for
    plotting. It does this in two ways:

    :param metrics: Store the metrics that we want to track
    :param unreplicate: Convert the metrics from a replicated
    :param stack: Stack the metrics in a list
    :return: A dictionary of metrics
    
    """
    if unreplicate:
        metrics = flax.jax_utils.unreplicate(metrics)
    metrics = jax.device_get(metrics)
    if stack:
        return jax.tree_map(lambda *args: np.stack(args), *metrics)
    else:
        return {key: float(val) for key, val in metrics.items()}


def tree_path_to_string(path, sep=None):
    """
    The tree_path_to_string function takes a tree path and returns a string representation of it.

    :param path: Specify the path of the tree
    :param sep: Join the keys with a separator
    :return: A tuple of strings
    
    """
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)


def flatten_tree(xs, is_leaf=None, sep=None):
    """
    The flatten_tree function takes a nested structure of arrays and returns a
    dictionary mapping from string keys to the corresponding array values. The
    string keys are derived from the tree path to each value, with `sep` used as
    the separator between levels in the tree. For example:

    :param xs: Store the tree structure
    :param is_leaf: Determine if a node is a leaf
    :param sep: Specify the separator between each key in the path
    :return: A dict of flattened tree paths to values
    
    """
    flattened, _ = jax.tree_util.tree_flatten_with_path(xs, is_leaf=is_leaf)
    output = {}
    for key, val in flattened:
        output[tree_path_to_string(key, sep=sep)] = val
    return output


def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree, *rest,
        is_leaf=is_leaf
    )


def match_partition_rules(rules, params):
    """ Returns a pytree of PartitionSpec according to rules. Supports handling
        Flax TrainState and Optax optimizer state.
    """

    def get_partition_spec(name, leaf):
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            """ Don't partition scalar values. """
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {name}')

    return named_tree_map(get_partition_spec, params, sep='/')


def get_weight_decay_mask(exclusions):
    """ Return a weight decay mask function that computes the pytree masks
        according to the given exclusion rules.
    """

    def decay(name, _):
        for rule in exclusions:
            if re.search(rule, name) is not None:
                return False
        return True

    def weight_decay_mask(params):
        return named_tree_map(decay, params, sep='/')

    return weight_decay_mask


def tree_apply(fns, tree):
    """
    The tree_apply function is a generalization of the map function.
    It takes two arguments: a pytree of functions and a pytree of values.
    The tree_apply function applies each function in the first argument to its corresponding value in the second argument,
    and returns a new pytree with these results.

    :param fns: Apply the functions to the tree
    :param tree: Apply the function to each element in the tree
    :return: A pytree of the same structure as the input
    """
    return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)


def create_mesh(
        axis_dims: Sequence[int] = (1, -1, 1, 1), axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"), backend=""
):
    """
    The create_mesh function creates a mesh object that can be used to shard arrays.

    :param axis_dims: Sequence[int]: Specify the dimensions of the mesh
    :param axis_names: Sequence[str]: Name the axes of the mesh
    :param backend: Specify the backend to use
    :return: A mesh object

    """
    array_devices = jax.numpy.ones((len(jax.devices() if backend == "" else jax.devices(backend)), 1))
    resh = array_devices.reshape(axis_dims).shape

    return jax.sharding.Mesh(
        create_device_mesh(resh), axis_names
    )
