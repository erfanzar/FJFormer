from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import (
    dataclass,
    field,
    fields,
    is_dataclass
)
from typing import ClassVar, Union
import warnings

import numpy as np
from jax.api_util import flatten_fun
import jax.interpreters.partial_eval as pe
from jax.core import Shape
from jax.tree_util import register_pytree_with_keys_class

from jax.typing import DTypeLike

import jax.numpy as jnp
from jax.dtypes import float0
import optax

from typing import Any, Optional, Tuple

from plum import (
    dispatch,
    parametric,
    Dispatcher,
    Function
)
from abc import ABC
from itertools import count

from functools import partial, wraps
from itertools import chain

import jax
from jax.api_util import flatten_fun_nokwargs
from jax import core
import jax.extend.linear_util as lu
from jax import tree_util


class ArrayValue(metaclass=ABCMeta):
    """Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()
    shape = None
    e_num_val = None
    is_registered_by_pJit = False


ArrayValue.register(jax.Array)

_dispatch = Dispatcher()

_primitive_ids = count()

COMMUTATIVE_OPS = frozenset([
    "add",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "eq",
    "max",
    "min",
    "mul",
    "ne",
])

ELEMENTWISE_UNOPS = frozenset([
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "bessel_i0e",
    "bessel_i1e",
    "cbrt",
    "ceil",
    "clz",
    "conj",
    "convert_element_type",
    "copy",
    "cos",
    "cosh",
    "digamma",
    "erf_inv",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "floor",
    "imag",
    "integer_pow",
    "is_finite",
    "lgamma",
    "log1p",
    "log",
    "logistic",
    "neg",
    "not",
    "population_count",
    "real",
    "reduce_precision",
    "round",
    "rsqrt",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
])

ELEMENTWISE_BINOPS = frozenset([
    "add",
    "and",
    "atan2",
    "complex",
    "div",
    "eq",
    "ge",
    "gt",
    "igamma_grad_a",
    "igamma",
    "igammac",
    "le",
    "lt",
    "max",
    "min",
    "mul",
    "ne",
    "nextafter",
    "or",
    "pow",
    "random_gamma_grad",
    "rem",
    "shift_left",
    "shift_right_arithmetic",
    "shift_right_logical",
    "sub",
    "xor",
])

REDUCTION_OPS = frozenset([
    "argmax",
    "argmin",
    "reduce_and",
    "reduce_max",
    "reduce_min",
    "reduce_or",
    "reduce_prod",
    "reduce_sum",
    "reduce_xor",
])

CUMULATIVE_REDUCTION_OPS = frozenset([
    "cumlogsumexp",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
])

_GENERAL = -2
_SPECIALIZED = -1


def _with_implicit_flat(fun: lu.WrappedFun) -> lu.WrappedFun:
    return _implicit_outer(_implicit_inner(fun))


@lu.transformation
def _implicit_outer(*in_vals):
    with core.new_main(ImplicitArrayTrace) as main:
        outs = yield (main, *in_vals), {}
        del main
    yield outs


@lu.transformation
def _implicit_inner(main, *in_vals):
    trace = main.with_cur_sublevel()
    in_tracers = [
        ImplicitArrayTracer(trace, val) if isinstance(val, ImplicitArray) else val
        for val in in_vals
    ]
    outs = yield in_tracers, {}
    out_vals = [trace.full_raise(t).value for t in outs]
    yield out_vals


def use_implicit_args(f):
    """
    Decorator which allows a function to accept arguments which subclass ImplicitArray, possibly
    including further ImplicitArray instances as children.
    Any number of arguments (including 0) may be ImplicitArrays.
    """

    @wraps(f)
    def implicit_f(*args, **kwargs):
        flat_args, in_tree = tree_flatten_with_implicit((args, kwargs))
        f_flat, out_tree = flatten_fun(lu.wrap_init(f), in_tree)
        f_wrapped = _with_implicit_flat(f_flat)
        outs_flat = f_wrapped.call_wrapped(*flat_args)
        return out_tree().unflatten(outs_flat)

    return implicit_f


def aux_field(metadata: Optional[Union[dict, Any]] = None, **kwargs):
    metadata = dict(metadata) if metadata else {}
    metadata["implicit_array_aux"] = True
    return field(metadata=metadata, **kwargs)


class UninitializedAval(Exception):
    def __init__(self, kind):
        super().__init__(_AVAL_ERROR_MESSAGE.format(kind))


class _AvalDescriptor:
    def __set_name__(self, owner, name):
        self._name = f"_{name}"

    def __get__(self, obj, owner=None):
        if obj is None:
            return None
        result = getattr(obj, self._name, None)
        if result is None:
            raise UninitializedAval(kind=self._name[1:])
        return result

    def __set__(self, obj, value):
        setattr(obj, self._name, value)


_aval_discovery = ContextVar("aval_discovery", default=False)


@contextmanager
def _aval_discovery_context():
    token = _aval_discovery.set(True)
    try:
        yield
    finally:
        _aval_discovery.reset(token)


@dataclass
class _ImplicitArrayBase(ArrayValue, ABC):
    commute_ops: ClassVar[bool] = True
    default_shape: ClassVar[Optional[Shape]] = None
    default_dtype: ClassVar[Optional[DTypeLike]] = None

    shape: Optional[Shape] = aux_field(kw_only=True, default=None)
    dtype: DTypeLike = aux_field(kw_only=True, default=None)


@dataclass
class ImplicitArray(_ImplicitArrayBase):
    """
    Abstract class for representing an abstract array of a given shape/dtype without actually instantiating it.
    Subclasses must implement the materialize method, which defines the relationship between the implicit array
    and the value it represents. Subclasses are valid arguments to functions decorated with qax.use_implicit_args.

    All subclasses are automatically registered as pytrees using jax.tree_util.register_pytree_with_keys_class.
    Any dataclass attributes added will be included as children, unless they are decorated with qax.aux_field
    in which case they are passed as auxiliary data during flattening.

    The represented shape and dtype may be defined in any of the following ways:
        - Explicitly passing shape/dtype keyword arguments at initialization
        - Overriding the default_shape/default_dtype class variables
        - Overriding the compute_shape/compute_dtype methods, which are called during __post_init__
        - Overriding __post_init__ and manually setting shape/dtype before calling super().__post_init__
        - None of the above, in which case an shape/dtype will be inferred by by running jax.eval_shape()
          on the subclass"s materialize method.
    """

    shape = _AvalDescriptor()
    dtype = _AvalDescriptor()

    def __post_init__(self):
        try:
            aval = _get_materialization_aval(self)
        except UninitializedAval:
            # Materialization depends on currently uninitialized shape/dtype
            aval = None

        shape = None
        try:
            shape = self.shape
        except UninitializedAval as e:
            shape = self.shape = self.compute_shape()

        if aval is not None:
            if shape is None:
                self.shape = aval.shape
            elif shape != aval.shape:
                warnings.warn(f"ImplicitArray shape {shape} does not match materialization shape {aval.shape}")
        elif shape is None:
            raise UninitializedAval("shape")

        dtype = None
        try:
            dtype = self.dtype
        except UninitializedAval as e:
            dtype = self.dtype = self.compute_dtype()

        if dtype is None and aval is None:
            # We have a shape but not a dtype, try once again to infer the dtype
            aval = _get_materialization_aval(self)

        if aval is not None:
            if dtype is None:
                self.dtype = aval.dtype
            elif dtype != aval.dtype:
                warnings.warn(f"ImplicitArray dtype {dtype} does not match materialization dtype {aval.dtype}")
        elif dtype is None:
            raise UninitializedAval("dtype")

    def compute_shape(self):
        """
        Override this method if the subclass instance"s shape should be computed based on its other properties.
        Returns: shape
        """
        return self.default_shape

    def compute_dtype(self):
        """
        Override this method if the subclass instance"s dtype should be computed based on its other properties.
        Returns: dtype
        """
        return self.default_dtype

    @property
    def aval(self):
        return core.ShapedArray(self.shape, self.dtype)

    @classmethod
    def default_handler(cls, primitive, *args, params=None):
        if params is None:
            params = {}
        return materialize_handler(primitive, *args, params=params)

    @abstractmethod
    def materialize(self):
        pass

    def tree_flatten_with_keys(self):
        children = []
        aux_data = []
        for name, is_aux in _get_names_and_aux(self):
            try:
                value = getattr(self, name)
            except UninitializedAval:
                if not _aval_discovery.get():
                    raise
                value = None
            if is_aux:
                aux_data.append(value)
            else:
                children.append((name, value))

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        child_it = iter(children)
        aux_it = iter(aux_data)
        obj = cls.__new__(cls)
        for name, is_aux in _get_names_and_aux(cls):
            value = next(aux_it if is_aux else child_it)
            setattr(obj, name, value)

        return obj

    def handle_primitive(self, primitive, *args, params):
        handler = lu.wrap_init(partial(get_primitive_handler(primitive), primitive))
        use_params = params

        if len(args) == 2 and self.commute_ops:
            args, use_params = _maybe_swap_args(primitive.name, args, use_params)

        # maybe_kwargs = {"params": params} if params else {}
        flat_args, in_tree = flatten_one_implicit_layer((args, params))
        flat_handler, out_tree = flatten_fun(handler, in_tree)

        result = use_implicit_args(flat_handler.call_wrapped)(*flat_args)
        return jax.tree_util.tree_unflatten(out_tree(), result)

    def __init_subclass__(cls, commute_ops=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")
        core.pytype_aval_mappings[cls] = lambda x: x.aval
        register_pytree_with_keys_class(cls)
        return cls


def _get_names_and_aux(obj):
    for val in fields(obj):
        yield val.name, bool(val.metadata.get("implicit_array_aux"))


def _materialize_all(it):
    return [materialize_nested(val) if isinstance(val, ImplicitArray) else val for val in it]


def _maybe_swap_args(op_name, args, params):
    if isinstance(args[0], ImplicitArray):
        return args, params
    if op_name in COMMUTATIVE_OPS:
        return args[::-1], params

    return args, params


class ImplicitArrayTracer(core.Tracer):
    def __init__(self, trace, value):
        super().__init__(trace)
        self.value = value

    @property
    def aval(self):
        if isinstance(self.value, ImplicitArray):
            return self.value.aval
        return core.get_aval(self.value)

    def full_lower(self):
        if isinstance(self.value, ImplicitArray):
            return self

        return core.full_lower(self.value)


class ImplicitArrayTrace(core.Trace):
    pure = lift = lambda self, val: ImplicitArrayTracer(self, val)

    def process_primitive(self, primitive, tracers, params):
        outs = NotImplemented
        vals = [t.value for t in tracers]
        implicit_idx = next(i for i, v in enumerate(vals) if isinstance(v, ImplicitArray))

        # First try to handle the primitive using custom handlers
        outs = vals[implicit_idx].handle_primitive(primitive, *vals, params=params)

        if outs is NotImplemented:
            # For higher order primitives most users won"t implement custom
            # logic, so there shouldn"t be a warning
            if primitive.name in _default_handlers:
                outs = _default_handlers[primitive.name](primitive, *vals, params=params)
            else:
                warnings.warn(
                    f"Primitive {primitive.name} was not handled by class {vals[implicit_idx].__class__.__name__}, so implicit args will be materialized.")

        if outs is NotImplemented:
            outs = vals[implicit_idx].default_handler(primitive, *vals, params=params)

        if primitive.multiple_results:
            return [ImplicitArrayTracer(self, out) for out in outs]
        return ImplicitArrayTracer(self, outs)


def wrap_jaxpr(jaxpr, vals_with_implicits, return_closed=True):
    if isinstance(jaxpr, jax.core.ClosedJaxpr):
        literals = jaxpr.literals
        jaxpr = jaxpr.jaxpr
    else:
        literals = []

    wrapped_fn = lu.wrap_init(use_implicit_args(partial(core.eval_jaxpr, jaxpr)))
    flat_args, in_tree = jax.tree_util.tree_flatten((literals, *vals_with_implicits))
    flat_fn, out_tree = flatten_fun_nokwargs(wrapped_fn, in_tree)

    new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fn, [core.get_aval(v) for v in flat_args])

    ret = (jax.core.ClosedJaxpr(new_jaxpr, consts),) if return_closed else (new_jaxpr, consts)
    return *ret, flat_args, out_tree()


def _transform_jaxpr_output(jaxpr, jaxpr_args, orig_out_struct, out_transform):
    def eval_fn(literals, *args):
        output = use_implicit_args(partial(core.eval_jaxpr, jaxpr.jaxpr))(literals, *args)
        unflattened_output = orig_out_struct.unflatten(output)
        return out_transform(unflattened_output)

    wrapped = lu.wrap_init(eval_fn)

    flat_args, in_tree = jax.tree_util.tree_flatten((jaxpr.literals, *jaxpr_args))
    flat_fn, out_tree = flatten_fun_nokwargs(wrapped, in_tree)
    new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fn, [core.get_aval(v) for v in flat_args])

    return jax.core.ClosedJaxpr(new_jaxpr, consts), out_tree()


def _match_branches(branches, arg_vals):
    out_avals = []
    new_jaxprs = []
    flat_inputs = None
    branch_out_struct = None
    for branch in branches:
        new_jaxpr, flat_inputs, branch_out_struct = wrap_jaxpr(branch, arg_vals)
        new_jaxprs.append((new_jaxpr, branch_out_struct))
        out_avals.append(
            branch_out_struct.unflatten(
                jax.eval_shape(
                    partial(core.eval_jaxpr, new_jaxpr.jaxpr), new_jaxpr.literals, *flat_inputs
                )
            )
        )

    out_transforms = get_common_prefix_transforms(out_avals)
    new_branches = []
    out_struct = None
    for (new_jaxpr, orig_out_struct), transform in zip(new_jaxprs, out_transforms):
        new_jaxpr, out_struct = _transform_jaxpr_output(new_jaxpr, flat_inputs, orig_out_struct, transform)
        new_branches.append(new_jaxpr)

    return tuple(new_branches), out_struct, flat_inputs


def _handle_cond(primitive, *vals, params):
    cond_val, *arg_vals = vals
    subfuns, bind_params = primitive.get_bind_params(params)

    new_branches, out_struct, flat_inputs = _match_branches(params["branches"], arg_vals)
    bind_params["branches"] = new_branches
    bind_params["linear"] = _broadcast_tuple(params["linear"], arg_vals)

    outs = primitive.bind(*subfuns, cond_val, *flat_inputs, **bind_params)
    return jax.tree_util.tree_unflatten(out_struct, outs)


def _handle_remat2(primitive, *vals, params):
    subfuns, bind_params = primitive.get_bind_params(params)
    new_jaxpr, consts, flat_inputs, out_tree = wrap_jaxpr(bind_params["jaxpr"], vals, return_closed=False)
    new_jaxpr = pe.convert_constvars_jaxpr(new_jaxpr)
    bind_params["jaxpr"] = new_jaxpr
    outs = primitive.bind(*subfuns, *consts, *flat_inputs, **bind_params)
    return jax.tree_util.tree_unflatten(out_tree, outs)


def _handle_pjit(primitive, *vals, params):
    new_jaxpr, flat_inputs, out_tree = wrap_jaxpr(params["jaxpr"], vals)
    donated_invars = _broadcast_tuple(params["donated_invars"], vals)
    in_shardings = _broadcast_tuple(params["in_shardings"], vals)
    out_shardings = _broadcast_tuple(params["out_shardings"], out_tree)

    subfuns, bind_params = primitive.get_bind_params(params)
    bind_params["jaxpr"] = new_jaxpr
    bind_params["donated_invars"] = donated_invars
    bind_params["in_shardings"] = in_shardings
    bind_params["out_shardings"] = out_shardings
    outs = primitive.bind(*subfuns, *flat_inputs, **bind_params)
    return jax.tree_util.tree_unflatten(out_tree, outs)


_default_handlers = {
    "cond": _handle_cond,
    "remat2": _handle_remat2,
    "pjit": _handle_pjit,
}


def materialize_handler(primitive, *vals, params):
    vals = _materialize_all(vals)
    subfuns, bind_params = primitive.get_bind_params(params)
    result = use_implicit_args(primitive.bind)(*subfuns, *vals, **bind_params)
    return result


def _broadcast_tuple(t, trees):
    if isinstance(trees, jax.tree_util.PyTreeDef):
        trees = jax.tree_util.tree_unflatten(trees, range(trees.num_leaves))
    assert len(t) == len(trees)
    return tuple(chain.from_iterable(
        (tuple_val for _ in jax.tree_util.tree_leaves(tree))
        for tuple_val, tree in zip(t, trees)
    ))


def _get_materialization_aval(imp_arr):
    with _aval_discovery_context(), _filter_materialization_warnings():
        result = jax.eval_shape(
            partial(materialize_nested, full=True),
            imp_arr
        )
    return result


@contextmanager
def _filter_materialization_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Primitive.*was not handled")
        yield


_AVAL_ERROR_MESSAGE = (
    "{} was not set during initialization. Shape and dtype may be set by:"
    "\n\t1. Directly passing them as keyword arguments to ImplicitArray instances"
    "\n\t2. Overriding the default_shape/default_dtype class attributes"
    "\n\t3. Overriding the compute_shape/compute_dtype methods"
    "\n\t4. Overriding __post_init__ and setting their values there"
    "\n\t5. None of the above, in which case `materialize()` will be called in an attempt to infer them."
    " If their values are required in order to compute the materialization this will be unsuccessful."
)


class _EmptyNodeCls:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


EmptyNode = _EmptyNodeCls()

tree_util.register_pytree_node(
    _EmptyNodeCls,
    lambda node: ((), None),
    lambda _, __: EmptyNode
)


def combine_leaf_predicate(base_fn, is_leaf):
    @wraps(base_fn)
    def new_fn(*args, new_is_leaf=None):
        if new_is_leaf is None:
            combined_is_leaf = is_leaf
        else:
            def combined_is_leaf(arg):
                return is_leaf(arg) or new_is_leaf(arg)
        return base_fn(*args, is_leaf=combined_is_leaf)

    return new_fn


def leaf_predicate(x):
    return isinstance(x, (ImplicitArray, _EmptyNodeCls))


tree_map_with_implicit = combine_leaf_predicate(jax.tree_map, leaf_predicate)
tree_map_with_path_with_implicit = combine_leaf_predicate(tree_util.tree_map_with_path, leaf_predicate)
tree_flatten_with_implicit = combine_leaf_predicate(tree_util.tree_flatten, leaf_predicate)
tree_flatten_with_path_with_implicit = combine_leaf_predicate(tree_util.tree_flatten_with_path, leaf_predicate)
tree_leaves_with_implicit = combine_leaf_predicate(tree_util.tree_leaves, leaf_predicate)
tree_structure_with_implicit = combine_leaf_predicate(tree_util.tree_structure, leaf_predicate)


def flatten_one_implicit_layer(tree):
    def is_leaf_below_node(node, x):
        return isinstance(x, ImplicitArray) and x is not node

    def replace_subtree_implicits(node):
        return tree_util.tree_map(lambda _: 1, node, is_leaf=partial(is_leaf_below_node, node))

    prototype = tree_map_with_implicit(replace_subtree_implicits, tree)
    struct = tree_util.tree_structure(prototype)

    leaves = tree_leaves_with_implicit(tree)
    leaves = list(chain.from_iterable(
        tree_util.tree_leaves(leaf, is_leaf=partial(is_leaf_below_node, leaf))
        if isinstance(leaf, ImplicitArray) else
        [leaf] for leaf in leaves
    ))
    return leaves, struct


def implicit_depth(tree):
    leaves = tree_leaves_with_implicit(tree)
    depth = 0
    while True:
        next_leaves = []
        any_implicit = False
        for leaf in leaves:
            if not isinstance(leaf, ImplicitArray):
                continue
            any_implicit = True
            next_leaves.extend(flatten_one_implicit_layer(leaf)[0])

        if not any_implicit:
            return depth

        depth += 1
        leaves = next_leaves


def _map_leaves_with_implicit_path(f, leaves, is_leaf, path_prefix=()):
    mapped_leaves = []
    for idx, leaf in enumerate(leaves):
        path = path_prefix + (idx,)
        if not isinstance(leaf, ImplicitArray) or is_leaf(path, leaf):
            mapped_leaves.append(f(leaf))
            continue

        subtree, substruct = flatten_one_implicit_layer(leaf)
        mapped_subtree = _map_leaves_with_implicit_path(
            f,
            subtree,
            is_leaf=is_leaf,
            path_prefix=path
        )
        mapped_leaves.append(tree_util.tree_unflatten(substruct, mapped_subtree))
    return mapped_leaves


def _get_pruning_transform(tree, materialization_paths):
    if not materialization_paths:
        return lambda x: x

    def is_leaf(path, leaf):
        return path in materialization_paths

    def materialize_subtrees(tree):
        leaves, struct = tree_flatten_with_implicit(tree)

        mapped_leaves = _map_leaves_with_implicit_path(partial(materialize_nested, full=True), leaves, is_leaf)
        return tree_util.tree_unflatten(struct, mapped_leaves)

    return materialize_subtrees


def get_common_prefix_transforms(trees):
    """
    Given an iterable of pytrees which have the same structure after all
    ImplicitArray instances are materialized, return a list of callables
    which will transform each tree into the largest common structure
    obtainable via materialization of ImplicitArrays.
    """
    if len(trees) <= 1:
        return [lambda x: x for _ in trees]

    all_leaves, structures = zip(*(tree_flatten_with_implicit(tree) for tree in trees))
    post_materialization_avals = [core.get_aval(leaf) for leaf in all_leaves[0]]
    for i, (leaves, structure) in enumerate(zip(all_leaves[1:], structures[1:]), 1):
        if structure != structures[0]:
            raise ValueError('Trees do not have the same structure after materialization')

        for leaf, expected_aval in zip(leaves, post_materialization_avals):
            aval = core.get_aval(leaf)
            if not (aval.shape == expected_aval.shape and aval.dtype == expected_aval.dtype):
                raise ValueError(
                    f'Trees do not have the same avals after materialization. Tree 0: {expected_aval}, Tree {i}: {aval}'
                )

    # Stack will contain tuples of (path, nodes)
    # path = a sequence of integers specifying which child
    # was taken at each _flatten_one_implicit_layer call
    # or the first flatten_with_implicit call
    # nodes = one node from each tree
    stack = []

    all_leaves = []
    for tree in trees:
        all_leaves.append(tree_leaves_with_implicit(tree))

    for i, nodes in enumerate(zip(*all_leaves)):
        stack.append(((i,), nodes))

    materialization_paths = set()
    while stack:
        path_prefix, nodes = stack.pop()
        if not any(isinstance(node, ImplicitArray) for node in nodes):
            continue

        all_leaves, all_structures = zip(*(
            flatten_one_implicit_layer(node) for node in nodes
        ))
        node_structures = set(all_structures)
        if len(node_structures) > 1:
            materialization_paths.add(path_prefix)
            continue

        aval_diff = False
        for leaves in zip(*all_leaves):
            first_aval = core.get_aval(leaves[0])
            shape = first_aval.shape
            dtype = first_aval.dtype
            for leaf in leaves[1:]:
                aval = core.get_aval(leaf)
                if not (aval.shape == shape and aval.dtype == dtype):
                    materialization_paths.add(path_prefix)
                    aval_diff = True
            if aval_diff:
                break

        if aval_diff:
            continue

        for i, leaf_group in enumerate(zip(*all_leaves)):
            stack.append((path_prefix + (i,), leaf_group))

    return [_get_pruning_transform(tree, materialization_paths) for tree in trees]


def materialize_nested(implicit_arr, full=False):
    """
    Materialize an ImplicitArray instance, handling the case where implicit_arr.materialize()
    involves further ImplicitArray instances.
    Arguments:
        implicit_arr: An ImplicitArray instance
        full: If True, repeatedly materialize until the result is a concrete array
    Returns:
        The materialized array
    """
    while isinstance(implicit_arr, ImplicitArray):
        wrapped = lu.wrap_init(type(implicit_arr).materialize)
        flat, in_tree = flatten_one_implicit_layer((implicit_arr,))
        flat_fn, out_tree = flatten_fun_nokwargs(wrapped, in_tree)
        out_flat = use_implicit_args(flat_fn.call_wrapped)(*flat)
        implicit_arr = jax.tree_util.tree_unflatten(out_tree(), out_flat)

        if not full:
            break

    return implicit_arr


def get_lax_primitive_by_name(name):
    return getattr(jax.lax, f"{name}_p")


def get_primitive_handler(primitive):
    if isinstance(primitive, str):
        primitive = get_lax_primitive_by_name(primitive)
    handler = _dispatch.functions.get(primitive)
    if handler is None:
        def _not_impl_handler(primitive: jax.core.Primitive, *args, **kwargs):
            return NotImplemented

        _not_impl_handler.__doc__ = "Default handler for {primitive.name}"
        handler = Function(_not_impl_handler)
        handler.register(_not_impl_handler, precedence=-1e9)
        handler.__name__ = f"{primitive.name}_{next(_primitive_ids)}"
        _dispatch.functions[primitive] = handler
    return handler


def primitive_handler(primitives, precedence=0):
    if isinstance(primitives, (str, jax.core.Primitive)):
        primitives = [primitives]

    def decorator(fn):
        for primitive in primitives:
            handler = get_primitive_handler(primitive)
            handler.register(fn, precedence=precedence)

    return decorator


def default_handler(primitive, *args, **params):
    subfuns, bind_params = primitive.get_bind_params(params)
    return primitive.bind(*subfuns, *args, **bind_params)


class _ComplementMeta(type):
    def __instancecheck__(self, x):
        a, b = self.type_parameter
        return (
                a is None or (
                isinstance(x, a) and not isinstance(x, b)
        )
        )


@parametric
class Complement(metaclass=_ComplementMeta):
    """
    Relative complement
    I.e. Complement[A, B] = A - B
    """

    @classmethod
    @dispatch
    def __init_type_parameter__(
            cls,
            a: Optional[Any],
            b: Optional[Any],
    ):
        return a, b

    @classmethod
    @dispatch
    def __le_type_parameter__(
            cls,
            left: Tuple[Optional[Any], Optional[Any]],
            right: Tuple[Optional[Any], Optional[Any]],
    ):
        a_left, b_left = left
        a_right, b_right = right

        return issubclass(a_left, a_right) and issubclass(b_right, b_left)


def vmap_all_but_one(f, axis, out_ndim=0):
    """
    Repeatedly calls vmap to map over all axes except for `axis.`
    All args will be mapped on the same dimensions.
    """

    @wraps(f)
    def inner(*args):
        n_dim = args[0].ndim
        if axis >= n_dim:
            raise ValueError(f'Axis {axis} is out of bounds for array of dimension {n_dim}')
        fn = f
        vmap_dim = 1
        out_dim = out_ndim
        for i in reversed(range(n_dim)):
            if i == axis:
                vmap_dim = 0
                out_dim = 0
            else:
                fn = jax.vmap(fn, vmap_dim, out_dim)
        return fn(*args)

    return inner


def freeze_subtrees(optimizer: optax.GradientTransformation, label_fn, use_scalar_zeros=False):
    """
    Utility which wraps an optimizer such that subtrees specified by
    label_fn will receive zeros as updates.
    Subtrees to be frozen should be labeled with "freeze"
    and all other subtrees should be labeled with "train"
    """
    multi_transformed_optimizer = optax.multi_transform(
        {
            'freeze': set_to_zero_scalar() if use_scalar_zeros else optax.set_to_zero(),
            'train': optimizer
        },
        label_fn
    )

    def new_update(grads, opt_state, params):
        def map_float0(param, grad):
            if grad.dtype == float0:
                return jnp.zeros((), param.dtype) if use_scalar_zeros else jnp.zeros_like(param)
            return grad

        fixed_grads = jax.tree_map(map_float0, params, grads)
        return multi_transformed_optimizer.update(fixed_grads, opt_state, params)

    return optax.GradientTransformation(
        multi_transformed_optimizer.init,
        new_update
    )


def freeze_keys(optimizer: optax.GradientTransformation, arr_type, keys,
                use_scalar_zeros=False) -> optax.GradientTransformation:
    keys = set(keys)

    def label_leaf(leaf):
        if not isinstance(leaf, arr_type):
            return 'train'

        children, aux_data = leaf.tree_flatten_with_keys()
        labels = ['freeze' if key in keys else 'train' for key, _ in children]
        struct = leaf.tree_unflatten(aux_data, labels)
        return struct

    def label_fn(root):
        return jax.tree_map(label_leaf, root, is_leaf=lambda x: isinstance(x, arr_type))

    return freeze_subtrees(optimizer, label_fn, use_scalar_zeros=use_scalar_zeros)


def apply_updates(params: optax.Params, updates: optax.Updates) -> optax.Params:
    """
    Like optax.apply_updates, but updates can be SymbolicConstant instances
    """
    updates_flat, update_struct = tree_util.tree_flatten(updates, is_leaf=lambda x: isinstance(x, SymbolicConstant))
    semi_flat_params = update_struct.flatten_up_to(params)

    updated_flat = use_implicit_args(optax.apply_updates)(semi_flat_params, updates_flat)
    updated = update_struct.unflatten(updated_flat)
    return updated


def set_to_zero_scalar() -> optax.GradientTransformation:
    """
    Returns a gradient transformation that sets all gradients to 0 in order to
    make downstream constant folding cheaper.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        return jax.tree_map(lambda x: jnp.zeros((), x.dtype), updates), state

    return optax.GradientTransformation(init_fn, update_fn)


# WARNING: This file is obviously super incomplete, and is
# currently just for convenience in testing.


def _get_shape_dtype(x, shape, dtype):
    if shape is None:
        shape = np.shape(x)
    else:
        shape = jax.core.canonicalize_shape(shape)

    if dtype is None:
        jax.lax.dtype(x)
    return shape, dtype


def _out_shape_dtype(primitive, *args, **kwargs):
    out_aval = jax.eval_shape(
        partial(default_handler, primitive, **kwargs),
        *(jax.core.get_aval(x) for x in args)
    )
    return jax.tree_map(
        lambda x: (x.shape, x.dtype),
        out_aval
    )


def symbolic_zero_like(x, shape=None, dtype=None):
    dtype = jax.lax.dtype(x) if dtype is None else dtype
    return symbolic_full_like(x, 0, shape=shape, dtype=dtype)


def symbolic_full_like(x, fill_value, shape=None, dtype=None):
    shape, _ = _get_shape_dtype(x, shape, None)
    if dtype is None:
        dtype = jax.lax.dtype(fill_value)

    return SymbolicConstant(fill_value, shape=shape, dtype=dtype)


@dataclass
class SymbolicConstant(ImplicitArray):
    value: Any = aux_field()
    weak_type: bool = aux_field(default=False)

    # commute_ops: ClassVar[bool] = True
    # default_shape: ClassVar[Optional[Shape]] = None
    # default_dtype: ClassVar[Optional[DTypeLike]] = None
    # shape: Optional[Shape] = aux_field(kw_only=True, default=None)
    # dtype: DTypeLike = aux_field(kw_only=True, default=None)
    def __post_init__(self):
        super().__post_init__()
        with jax.ensure_compile_time_eval():
            self.value = jnp.asarray(self.value, dtype=self.dtype)

    def compute_dtype(self):
        return jax.lax.dtype(self.value)

    def materialize(self):
        return jnp.full(self.shape, self.value, dtype=self.dtype)

    def copy(self):
        return jax.tree_map(lambda x: x, self)


@use_implicit_args
def broadcast_to(val, shape):
    return jnp.broadcast_to(val, shape)


@use_implicit_args
def astype(val, dtype):
    return val.astype(dtype)


@primitive_handler([
    "reshape",
    "broadcast_in_dim",
    "reduce_min",
    "reduce_max",
    "reduce_or",
    "reduce_and"
])
def unchanged_value_op(primitive, sym: SymbolicConstant, **kwargs):
    out_shape, out_dtype = _out_shape_dtype(primitive, sym, **kwargs)
    return SymbolicConstant(sym.value, shape=out_shape, dtype=out_dtype)


def _op_and_reshape(primitive, lhs, rhs, flip=False):
    """
    Close over one arg so we can do math at tracing time, but let the other one get traced
    """
    if flip:
        lhs, rhs = (rhs, lhs)

    @use_implicit_args
    def inner(arg):
        other = rhs
        if flip:
            arg, other = (other, arg)

        result = default_handler(primitive, arg, other)
        return result

    return inner(rhs)


def special_case_binop(name, identity=None, annihilator=None, flip=False):
    lhs_type = SymbolicConstant
    rhs_type = Complement[ArrayValue, SymbolicConstant]
    if flip:
        lhs_type, rhs_type = rhs_type, lhs_type

    @primitive_handler(name, precedence=_SPECIALIZED)
    def handler(primitive, lhs: lhs_type, rhs: rhs_type, **kwargs):
        out_shape, out_dtype = _out_shape_dtype(primitive, lhs, rhs, **kwargs)
        with jax.ensure_compile_time_eval():
            if lhs.value == identity:
                return broadcast_to(astype(rhs, out_dtype), out_shape)

            if lhs.value == annihilator:
                return SymbolicConstant(lhs.value, shape=out_shape, dtype=out_dtype)

            return _op_and_reshape(primitive, lhs.value, rhs)


special_case_binop("add", identity=0)
special_case_binop("mul", identity=1, annihilator=0)
special_case_binop("and", annihilator=0)
special_case_binop("or", identity=0)
special_case_binop("xor", identity=0)

special_case_binop("sub", identity=0, flip=True)
special_case_binop("div", identity=1, flip=True)
special_case_binop("exp", identity=1, flip=True)

special_case_binop("min", identity=float("inf"), annihilator=float("-inf"))
special_case_binop("max", identity=float("-inf"), annihilator=float("inf"))


def eval_default_handler(primitive, *args, **kwargs):
    with jax.ensure_compile_time_eval():
        result = primitive.bind(*args, **kwargs)
    return result


@primitive_handler(ELEMENTWISE_UNOPS, precedence=_GENERAL)
def handle_unop(primitive, sym: SymbolicConstant, **kwargs):
    print(f"Handling {primitive} with {sym}")
    new_val = eval_default_handler(primitive, sym.value, **kwargs)
    return symbolic_full_like(sym, new_val)


@primitive_handler(ELEMENTWISE_BINOPS, precedence=_GENERAL)
def handle_binop(primitive, lhs: SymbolicConstant, rhs: SymbolicConstant, **kwargs):
    out_shape, out_dtype = _out_shape_dtype(primitive, lhs, rhs, **kwargs)
    new_val = eval_default_handler(primitive, lhs.value, rhs.value, **kwargs)
    return symbolic_full_like(lhs, new_val, shape=out_shape, dtype=out_dtype)


@primitive_handler(["reduce_sum", "reduce_prod"])
def reduce_sum(primitive, sym: SymbolicConstant, *, axes):
    out_shape, out_dtype = _out_shape_dtype(primitive, sym, axes=axes)
    with jax.ensure_compile_time_eval():
        if sym.value == 0:
            return SymbolicConstant(0, shape=out_shape, dtype=out_dtype)

        orig_size = np.prod(sym.shape)
        new_size = np.prod(out_shape)

        n_combined = orig_size // new_size

        new_val = sym.value
        if primitive.name == "reduce_sum":
            new_val = new_val * n_combined
        else:
            new_val = new_val ** n_combined

    return SymbolicConstant(new_val, shape=out_shape, dtype=out_dtype)


@primitive_handler("select_n")
def handle_select_n(primitive, cond_val, *arg_vals: SymbolicConstant):
    if len(set(val.value.item() for val in arg_vals)) != 1:
        return NotImplemented

    out_shape, out_dtype = _out_shape_dtype(primitive, cond_val, *arg_vals)
    return SymbolicConstant(arg_vals[0].value, shape=out_shape, dtype=out_dtype)
