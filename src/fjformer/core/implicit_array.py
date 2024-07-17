"""
This module implements the ImplicitArray class and related functionality for representing
abstract arrays without instantiation. It provides mechanisms for lazy evaluation and
efficient handling of array operations in JAX.

Key components:
- ImplicitArray: Abstract base class for symbolic array representations
- primitive_handler: Decorator for registering custom primitive handlers
- implicit_compact: Decorator for functions to accept ImplicitArray arguments
"""

import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from functools import partial, wraps
from itertools import chain, count
from typing import Any, Callable, ClassVar, Optional, Tuple

import jax
import jax.extend.linear_util as lu
import jax.interpreters.partial_eval as pe
from jax import core, tree_util
from jax._src.typing import DTypeLike, Shape
from jax.api_util import flatten_fun, flatten_fun_nokwargs
from jax.tree_util import register_pytree_with_keys_class
from plum import Dispatcher, Function

from fjformer.core.errors import UninitializedAval

_dispatch = Dispatcher()
_primitive_ids = count()
warnings.filterwarnings(
    "ignore", message="Could not resolve the type hint of `~B`", module="plum.type"
)
warnings.filterwarnings(
    "ignore", message="Could not resolve the type hint of `~A`", module="plum.type"
)


class ArrayValue(ABC):
    pass


ArrayValue.register(jax.Array)


def get_lax_primitive_by_name(name: str) -> jax.core.Primitive:
    """Get a JAX LAX primitive by its name."""
    return getattr(jax.lax, f"{name}_p")


def get_primitive_handler(primitive):
    """Get or create a handler for a given primitive."""
    if isinstance(primitive, str):
        primitive = get_lax_primitive_by_name(primitive)
    handler = _dispatch.functions.get(primitive)
    if handler is None:

        def _not_impl_handler(primitive: jax.core.Primitive, *args, **kwargs):
            return NotImplemented

        _not_impl_handler.__doc__ = f"Default handler for {primitive.name}"
        handler = Function(_not_impl_handler)
        handler.register(_not_impl_handler, precedence=-1e9)
        handler.__name__ = f"{primitive.name}_{next(_primitive_ids)}"
        _dispatch.functions[primitive] = handler
    return handler


def primitive_handler(primitives, precedence=0):
    """Decorator to register a handler for one or more primitives."""
    if isinstance(primitives, (str, jax.core.Primitive)):
        primitives = [primitives]

    def decorator(fn):
        for primitive in primitives:
            handler = get_primitive_handler(primitive)
            handler.register(fn, precedence=precedence)

    return decorator


COMMUTATIVE_OPS = frozenset(
    [
        "add",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "eq",
        "max",
        "min",
        "mul",
        "ne",
    ]
)

ELEMENTWISE_UNOPS = frozenset(
    [
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
    ]
)

ELEMENTWISE_BINOPS = frozenset(
    [
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
    ]
)

REDUCTION_OPS = frozenset(
    [
        "argmax",
        "argmin",
        "reduce_and",
        "reduce_max",
        "reduce_min",
        "reduce_or",
        "reduce_prod",
        "reduce_sum",
        "reduce_xor",
    ]
)

CUMULATIVE_REDUCTION_OPS = frozenset(
    [
        "cumlogsumexp",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
    ]
)


class _EmptyNodeCls:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


EmptyNode = _EmptyNodeCls()

tree_util.register_pytree_node(
    _EmptyNodeCls, lambda node: ((), None), lambda _, __: EmptyNode
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


tree_map_with_implicit = combine_leaf_predicate(jax.tree_util.tree_map, leaf_predicate)
tree_map_with_path_with_implicit = combine_leaf_predicate(
    tree_util.tree_map_with_path, leaf_predicate
)
tree_flatten_with_implicit = combine_leaf_predicate(
    tree_util.tree_flatten, leaf_predicate
)
tree_flatten_with_path_with_implicit = combine_leaf_predicate(
    tree_util.tree_flatten_with_path, leaf_predicate
)
tree_leaves_with_implicit = combine_leaf_predicate(
    tree_util.tree_leaves, leaf_predicate
)
tree_structure_with_implicit = combine_leaf_predicate(
    tree_util.tree_structure, leaf_predicate
)


def flatten_one_implicit_layer(tree):
    def is_leaf_below_node(node, x):
        return isinstance(x, ImplicitArray) and x is not node

    def replace_subtree_implicits(node):
        return tree_util.tree_map(
            lambda _: 1, node, is_leaf=partial(is_leaf_below_node, node)
        )

    prototype = tree_map_with_implicit(replace_subtree_implicits, tree)
    struct = tree_util.tree_structure(prototype)

    leaves = tree_leaves_with_implicit(tree)
    leaves = list(
        chain.from_iterable(
            (
                tree_util.tree_leaves(leaf, is_leaf=partial(is_leaf_below_node, leaf))
                if isinstance(leaf, ImplicitArray)
                else [leaf]
            )
            for leaf in leaves
        )
    )
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
            f, subtree, is_leaf=is_leaf, path_prefix=path
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

        mapped_leaves = _map_leaves_with_implicit_path(
            partial(materialize_nested, full=True), leaves, is_leaf
        )
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
            raise ValueError(
                "Trees do not have the same structure after materialization"
            )

        for leaf, expected_aval in zip(leaves, post_materialization_avals):
            aval = core.get_aval(leaf)
            if not (
                aval.shape == expected_aval.shape and aval.dtype == expected_aval.dtype
            ):
                raise ValueError(
                    f"Trees do not have the same avals after materialization. Tree 0: {expected_aval}, Tree {i}: {aval}"
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

        all_leaves, all_structures = zip(
            *(flatten_one_implicit_layer(node) for node in nodes)
        )
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
        out_flat = implicit_compact(flat_fn.call_wrapped)(*flat)
        implicit_arr = jax.tree_util.tree_unflatten(out_tree(), out_flat)

        if not full:
            break

    return implicit_arr


def _with_implicit_flat(fun: lu.WrappedFun) -> lu.WrappedFun:
    """Wrap a function to handle implicit arrays."""
    f = _implicit_inner(fun)
    return _implicit_outer(f)


@lu.transformation
def _implicit_outer(*in_vals):
    """Outer transformation for implicit array handling."""
    with core.new_main(ImplicitArrayTrace) as main:
        outs = yield (main, *in_vals), {}
        del main
    yield outs


@lu.transformation
def _implicit_inner(main, *in_vals):
    """Inner transformation for implicit array handling."""
    trace = main.with_cur_sublevel()
    in_tracers = [
        ImplicitArrayTracer(trace, val) if isinstance(val, ImplicitArray) else val
        for val in in_vals
    ]
    outs = yield in_tracers, {}
    out_vals = [trace.full_raise(t).value for t in outs]
    yield out_vals


def implicit_compact(f: Callable) -> Callable:
    """
    A decorator that enables compact handling of ImplicitArray subclasses within a function.
    This allows for seamless integration of custom array types in JAX operations.

    This decorator can be used in combination with jax.jit for optimized execution.

    Args:
        f: The function to be decorated.

    Returns:
        A wrapped function that can handle both regular arrays and ImplicitArray instances.

    Example:
        >>> @jax.jit
        >>> @implicit_compact
        >>> def f(a, b):
        ...     return jnp.dot(a, b)

        >>> result = f(regular_array, regular_array)
        >>> implicit_result = f(implicit_array, implicit_or_normal_array)
    """

    @wraps(f)
    def implicit_f(*args, **kwargs):
        flat_args, in_tree = tree_flatten_with_implicit((args, kwargs))
        f_flat, out_tree = flatten_fun(lu.wrap_init(f), in_tree)
        f_wrapped = _with_implicit_flat(f_flat)
        outs_flat = f_wrapped.call_wrapped(*flat_args)
        return out_tree().unflatten(outs_flat)

    return implicit_f


def default_handler(primitive: Any, *args: Any, **params: Any) -> Any:
    """Default handler for primitives."""
    subfuns, bind_params = primitive.get_bind_params(params)
    return primitive.bind(*subfuns, *args, **bind_params)


def aux_field(metadata: Optional[dict] = None, **kwargs: Any) -> Any:
    """Create an auxiliary field for ImplicitArray subclasses."""
    metadata = dict(metadata) if metadata else {}
    metadata["implicit_array_aux"] = True
    return field(metadata=metadata, **kwargs)


class _AvalDescriptor:
    """Descriptor for lazy initialization of shape and dtype."""

    def __set_name__(self, owner: Any, name: str) -> None:
        self._name = f"_{name}"

    def __get__(self, obj: Any, owner: Any = None) -> Any:
        if obj is None:
            return None
        result = getattr(obj, self._name, None)
        if result is None:
            raise UninitializedAval(kind=self._name[1:])
        return result

    def __set__(self, obj: Any, value: Any) -> None:
        setattr(obj, self._name, value)


# Context variable for aval discovery
_aval_discovery = ContextVar("aval_discovery", default=False)


@contextmanager
def _aval_discovery_context():
    """Context manager for aval discovery."""
    token = _aval_discovery.set(True)
    try:
        yield
    finally:
        _aval_discovery.reset(token)


@dataclass
class _ImplicitArrayBase(ArrayValue, ABC):
    """Base class for ImplicitArray with common attributes."""

    commute_ops: ClassVar[bool] = True
    warn_on_materialize: ClassVar[bool] = True
    default_shape: ClassVar[Optional[Tuple[int, ...]]] = None
    default_dtype: ClassVar[Optional[Any]] = None

    shape: Optional[Tuple[int, ...]] = aux_field(kw_only=True, default=None)
    dtype: Any = aux_field(kw_only=True, default=None)


@dataclass
class ImplicitArray(_ImplicitArrayBase):
    """
    Abstract class for representing an abstract array without instantiation.

    Subclasses must implement the materialize method, which defines the relationship
    between the implicit array and the value it represents. Subclasses are valid
    arguments to functions decorated with core.implicit_compact.

    The represented shape and dtype may be defined in various ways:
    1. Explicitly passing shape/dtype keyword arguments at initialization
    2. Overriding the default_shape/default_dtype class variables
    3. Overriding the compute_shape/compute_dtype methods
    4. Overriding __post_init__ and manually setting shape/dtype
    5. Inferred by running jax.eval_shape() on the subclass's materialize method
    """

    shape = _AvalDescriptor()
    dtype = _AvalDescriptor()

    def __post_init__(self):
        """Initialize shape and dtype after instance creation."""
        try:
            aval = _get_materialization_aval(self)
        except UninitializedAval:
            aval = None

        shape = None
        try:
            shape = self.shape
        except UninitializedAval:
            shape = self.shape = self.compute_shape()

        if aval is not None:
            if shape is None:
                self.shape = aval.shape
            elif shape != aval.shape:
                warnings.warn(
                    f"ImplicitArray shape {shape} does not match materialization shape {aval.shape}"
                )
        elif shape is None:
            raise UninitializedAval("shape")

        dtype = None
        try:
            dtype = self.dtype
        except UninitializedAval:
            dtype = self.dtype = self.compute_dtype()

        if dtype is None and aval is None:
            aval = _get_materialization_aval(self)

        if aval is not None:
            if dtype is None:
                self.dtype = aval.dtype
            elif dtype != aval.dtype:
                warnings.warn(
                    f"ImplicitArray dtype {dtype} does not match materialization dtype {aval.dtype}"
                )
        elif dtype is None:
            raise UninitializedAval("dtype")

    def compute_shape(self) -> Optional[Shape]:
        """Compute the shape of the implicit array."""
        return self.default_shape

    def compute_dtype(self) -> Optional[DTypeLike]:
        """Compute the dtype of the implicit array."""
        return self.default_dtype

    @property
    def aval(self) -> core.ShapedArray:
        """Get the abstract value of the implicit array."""
        return core.ShapedArray(self.shape, self.dtype)

    @classmethod
    def default_handler(cls, primitive, *args, params=None):
        """Default handler for primitives."""
        if params is None:
            params = {}
        return materialize_handler(primitive, *args, params=params)

    @abstractmethod
    def materialize(self):
        """Materialize the implicit array into a concrete array."""
        pass

    def tree_flatten_with_keys(self):
        """Flatten the implicit array for PyTree operations."""
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
        """Unflatten the implicit array from PyTree operations."""
        child_it = iter(children)
        aux_it = iter(aux_data)
        obj = cls.__new__(cls)
        for name, is_aux in _get_names_and_aux(cls):
            value = next(aux_it if is_aux else child_it)
            setattr(obj, name, value)

        return obj

    def handle_primitive(self, primitive, *args, params):
        """Handle a primitive operation on the implicit array."""
        handler = lu.wrap_init(partial(get_primitive_handler(primitive), primitive))
        use_params = params

        if len(args) == 2 and self.commute_ops:
            args, use_params = _maybe_swap_args(primitive.name, args, use_params)

        flat_args, in_tree = flatten_one_implicit_layer((args, params))
        flat_handler, out_tree = flatten_fun(handler, in_tree)

        result = implicit_compact(flat_handler.call_wrapped)(*flat_args)
        return jax.tree_util.tree_unflatten(out_tree(), result)

    def __init_subclass__(cls, commute_ops=True, warn_on_materialize=True, **kwargs):
        """Initialize a subclass of ImplicitArray."""
        super().__init_subclass__(**kwargs)
        cls.commute_ops = commute_ops
        cls.warn_on_materialize = warn_on_materialize

        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")
        core.pytype_aval_mappings[cls] = lambda x: x.aval
        register_pytree_with_keys_class(cls)
        return cls


def _get_names_and_aux(obj):
    for val in fields(obj):
        yield val.name, bool(val.metadata.get("implicit_array_aux"))


def _materialize_all(it):
    return [
        materialize_nested(val) if isinstance(val, ImplicitArray) else val for val in it
    ]


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
        implicit_idx = next(
            i for i, v in enumerate(vals) if isinstance(v, ImplicitArray)
        )

        # First try to handle the primitive using custom handlers
        outs = vals[implicit_idx].handle_primitive(primitive, *vals, params=params)

        if outs is NotImplemented:
            # For higher order primitives most users won't implement custom
            # logic, so there shouldn't be a warning
            if primitive.name in _default_handlers:
                outs = _default_handlers[primitive.name](
                    primitive, *vals, params=params
                )
            else:
                implicit_cls = vals[implicit_idx].__class__
                if implicit_cls.warn_on_materialize:
                    warnings.warn(
                        f"Primitive {primitive.name} was not handled by class {implicit_cls.__name__}, so implicit args will be materialized."
                    )

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

    wrapped_fn = lu.wrap_init(implicit_compact(partial(core.eval_jaxpr, jaxpr)))
    flat_args, in_tree = jax.tree_util.tree_flatten((literals, *vals_with_implicits))
    flat_fn, out_tree = flatten_fun_nokwargs(wrapped_fn, in_tree)

    new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
        flat_fn, [core.get_aval(v) for v in flat_args]
    )

    ret = (
        (jax.core.ClosedJaxpr(new_jaxpr, consts),)
        if return_closed
        else (new_jaxpr, consts)
    )
    return *ret, flat_args, out_tree()


def _transform_jaxpr_output(jaxpr, jaxpr_args, orig_out_struct, out_transform):
    def eval_fn(literals, *args):
        output = implicit_compact(partial(core.eval_jaxpr, jaxpr.jaxpr))(
            literals, *args
        )
        unflattened_output = orig_out_struct.unflatten(output)
        return out_transform(unflattened_output)

    wrapped = lu.wrap_init(eval_fn)

    flat_args, in_tree = jax.tree_util.tree_flatten((jaxpr.literals, *jaxpr_args))
    flat_fn, out_tree = flatten_fun_nokwargs(wrapped, in_tree)
    new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
        flat_fn, [core.get_aval(v) for v in flat_args]
    )

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
                    partial(core.eval_jaxpr, new_jaxpr.jaxpr),
                    new_jaxpr.literals,
                    *flat_inputs,
                )
            )
        )

    out_transforms = get_common_prefix_transforms(out_avals)
    new_branches = []
    out_struct = None
    for (new_jaxpr, orig_out_struct), transform in zip(new_jaxprs, out_transforms):
        new_jaxpr, out_struct = _transform_jaxpr_output(
            new_jaxpr, flat_inputs, orig_out_struct, transform
        )
        new_branches.append(new_jaxpr)

    return tuple(new_branches), out_struct, flat_inputs


def _handle_cond(primitive, *vals, params):
    cond_val, *arg_vals = vals
    subfuns, bind_params = primitive.get_bind_params(params)

    new_branches, out_struct, flat_inputs = _match_branches(
        params["branches"], arg_vals
    )
    bind_params["branches"] = new_branches
    bind_params["linear"] = _broadcast_tuple(bind_params["linear"], arg_vals)

    outs = primitive.bind(*subfuns, cond_val, *flat_inputs, **bind_params)
    return jax.tree_util.tree_unflatten(out_struct, outs)


def _handle_remat2(primitive, *vals, params):
    subfuns, bind_params = primitive.get_bind_params(params)
    new_jaxpr, consts, flat_inputs, out_tree = wrap_jaxpr(
        bind_params["jaxpr"], vals, return_closed=False
    )
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


def _handle_scan(primitive, *vals, params):
    n_consts = params["num_consts"]
    n_carry = params["num_carry"]

    consts = vals[:n_consts]
    real_n_consts = len(jax.tree_util.tree_leaves(consts))

    carries = vals[n_consts : n_consts + n_carry]
    xs = vals[n_consts + n_carry :]

    if any(isinstance(c, ImplicitArray) for c in carries):
        warnings.warn("Not Supported Yet.")
        carries = _materialize_all(carries)

    sliced_xs = jax.tree_util.tree_map(partial(jax.eval_shape, lambda x: x[0]), xs)

    for x in sliced_xs:
        if isinstance(x, ImplicitArray):
            assert len(x._shape) > 0, "Attempted to scan over a scalar."
            x._shape = x._shape[1:]

    jaxpr = params["jaxpr"]
    new_jaxpr, _, out_tree = wrap_jaxpr(
        jaxpr=jaxpr,
        vals_with_implicits=(*consts, *carries, *sliced_xs),
        return_closed=True,
    )

    flat_inputs = jax.tree_util.tree_leaves((jaxpr.literals, *consts, *carries, *xs))

    subfuns, bind_params = primitive.get_bind_params(params)
    bind_params["jaxpr"] = new_jaxpr
    bind_params["num_consts"] = real_n_consts
    bind_params["num_carry"] = len(carries)
    bind_params["linear"] = _broadcast_tuple(params["linear"], vals)

    outs = primitive.bind(*subfuns, *flat_inputs, **bind_params)
    return jax.tree_util.tree_unflatten(out_tree, outs)


_default_handlers = {
    "cond": _handle_cond,
    "remat2": _handle_remat2,
    "pjit": _handle_pjit,
    "scan": _handle_scan,
}


def materialize_handler(primitive, *vals, params):
    vals = _materialize_all(vals)
    subfuns, bind_params = primitive.get_bind_params(params)
    result = implicit_compact(primitive.bind)(*subfuns, *vals, **bind_params)
    return result


def _broadcast_tuple(t, trees):
    if isinstance(trees, jax.tree_util.PyTreeDef):
        trees = jax.tree_util.tree_unflatten(trees, range(trees.num_leaves))
    assert len(t) == len(trees)
    return tuple(
        chain.from_iterable(
            (tuple_val for _ in jax.tree_util.tree_leaves(tree))
            for tuple_val, tree in zip(t, trees)
        )
    )


def _get_materialization_aval(imp_arr):
    with _aval_discovery_context(), _filter_materialization_warnings():
        result = jax.eval_shape(partial(materialize_nested, full=True), imp_arr)
    return result


@contextmanager
def _filter_materialization_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Primitive.*was not handled")
        yield


# Most Core part's of this package is Developed from orginal implimentation of QAX from davisyoshida
