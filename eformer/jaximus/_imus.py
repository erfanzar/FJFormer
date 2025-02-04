# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Copyright 2023 The Equinox Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Portions of this code are derived from the Equinox library
# (https://github.com/patrick-kidger/equinox)

import abc
import functools as ft
import itertools as it
import typing as tp
from collections.abc import Callable, Sequence

import chex
import jax
import jax._src
import jax.core as core
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import plum
from jax.custom_derivatives import SymbolicZero as SZ
from typing_extensions import TypeGuard

from ._core import Partial, PyTree, field, module_update_wrapper
from ._tree_util import combine, partition

CT = tp.TypeVar("CT", bound=Callable)


def is_array(element: tp.Any) -> bool:
	return isinstance(element, (np.ndarray, np.generic, jax.Array))


class _EmptyNodeCls:
	_instance = None

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance


EmptyNode = _EmptyNodeCls()

jax.tree_util.register_pytree_node(
	_EmptyNodeCls,
	lambda node: ((), None),
	lambda _, __: EmptyNode,
)


def combine_leaf_predicate(base_fn, is_leaf):
	@ft.wraps(base_fn)
	def new_fn(*args, new_is_leaf=None):
		if new_is_leaf is None:
			combined_is_leaf = is_leaf
		else:

			def combined_is_leaf(arg):
				return is_leaf(arg) or new_is_leaf(arg)

		return base_fn(*args, is_leaf=combined_is_leaf)

	return new_fn


def leaf_predicate(x):
	return isinstance(x, (Value, _EmptyNodeCls))


tree_map_with_implicit = combine_leaf_predicate(jax.tree_map, leaf_predicate)
tree_map_with_path_with_implicit = combine_leaf_predicate(
	jax.tree_util.tree_map_with_path, leaf_predicate
)
tree_flatten_with_implicit = combine_leaf_predicate(
	jax.tree_util.tree_flatten, leaf_predicate
)
tree_flatten_with_path_with_implicit = combine_leaf_predicate(
	jax.tree_util.tree_flatten_with_path, leaf_predicate
)
tree_leaves_with_implicit = combine_leaf_predicate(
	jax.tree_util.tree_leaves, leaf_predicate
)
tree_structure_with_implicit = combine_leaf_predicate(
	jax.tree_util.tree_structure, leaf_predicate
)


def flatten_one_implicit_layer(tree):
	def is_leaf_below_node(node, x):
		return isinstance(x, Value) and x is not node

	def replace_subtree_implicits(node):
		return jax.tree_util.tree_map(
			lambda _: 1,
			node,
			is_leaf=ft.partial(is_leaf_below_node, node),
		)

	prototype = tree_map_with_implicit(replace_subtree_implicits, tree)
	struct = jax.tree_util.tree_structure(prototype)

	leaves = tree_leaves_with_implicit(tree)
	leaves = list(
		it.chain.from_iterable(
			(
				jax.tree_util.tree_leaves(leaf, is_leaf=ft.partial(is_leaf_below_node, leaf))
				if isinstance(leaf, Value)
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
			if not isinstance(leaf, Value):
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
		if not isinstance(leaf, Value) or is_leaf(path, leaf):
			mapped_leaves.append(f(leaf))
			continue

		subtree, substruct = flatten_one_implicit_layer(leaf)
		mapped_subtree = _map_leaves_with_implicit_path(
			f, subtree, is_leaf=is_leaf, path_prefix=path
		)
		mapped_leaves.append(jax.tree_util.tree_unflatten(substruct, mapped_subtree))
	return mapped_leaves


_rules: dict[core.Primitive, plum.Function] = {}


def register(
	primitive: tp.Union[core.Primitive, str],
	*,
	precedence: int = 0,
) -> Callable[[CT], CT]:
	if isinstance(primitive, str):
		lac = getattr(jax.lax, f"{primitive}_p", None)
		if lac is None:
			raise ValueError(f"couldn't verify given string primitive {primitive}_p")
		primitive = lac

	def _register(rule: CT) -> CT:
		try:
			existing_rule = _rules[primitive]
		except KeyError:

			def existing_rule():
				raise AssertionError()

			existing_rule.__name__ = f"{primitive}_dispatcher"
			existing_rule.__qualname__ = f"{primitive}_dispatcher"
			existing_rule = plum.Dispatcher().abstract(existing_rule)

			_rules[primitive] = existing_rule
		existing_rule.dispatch(rule, precedence=precedence)
		return rule

	return _register


class _CustomTracer(core.Tracer):
	__slots__ = ("value",)

	def __init__(self, trace: "_Tracer", value: "Value") -> None:
		assert _is_value(value)
		self._trace = trace
		self.value = value

	@property
	def aval(self):
		return self.value.aval()

	def full_lower(self):
		if isinstance(self.value, _DenseArrayValue):
			return core.full_lower(self.value.array)
		else:
			return self


def _default_process(
	primitive: core.Primitive,
	values: Sequence[tp.Union[chex.Array, "Value"]],
	params,
):
	arrays: list[chex.Array] = []
	for x in values:
		if _is_value(x):
			arrays.append(x.materialize())
		elif is_array(x):
			arrays.append(tp.cast(chex.Array, x))
		else:
			arrays.append(x)
	subfuns, bind_params = primitive.get_bind_params(params)
	return primitive.bind(*subfuns, *arrays, **bind_params)


def _wrap_if_array(x: tp.Union[chex.Array, "Value"]) -> "Value":
	if is_array(x):
		return _DenseArrayValue(tp.cast(chex.Array, x))
	else:
		return tp.cast(Value, x)


class _Tracer(core.Trace[_CustomTracer]):
	def __init__(self, parent_trace, tag):
		self.tag = tag
		self.parent_trace = parent_trace

	def to_value(self, val):
		if isinstance(val, _CustomTracer) and val._trace.tag is self.tag:
			return val.value
		return _DenseArrayValue(val)

	def process_primitive(self, primitive, tracers, params):
		values = [self.to_value(t) for t in tracers]
		values = tuple(x.array if isinstance(x, _DenseArrayValue) else x for x in values)
		try:
			rule = _rules[primitive]
		except KeyError:
			with core.set_current_trace(self.parent_trace):
				out = _default_process(primitive, values, params)
		else:
			with core.set_current_trace(self.parent_trace):
				try:
					method, _ = rule.resolve_method(values)
				except plum.NotFoundLookupError:
					out = _default_process(primitive, values, params)
				else:
					out = method(*values, **params)
		if primitive.multiple_results:
			out = [_CustomTracer(self, _wrap_if_array(x)) for x in out]
		else:
			out = _CustomTracer(self, _wrap_if_array(out))
		return out

	def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *, symbolic_zeros):
		in_values = [self.to_value(t) for t in tracers]
		in_leaves, in_treedef = tu.tree_flatten(in_values)
		fun, out_treedef1 = _custom_jvp_fun_wrap(fun, self.tag, in_treedef)
		jvp, out_treedef2 = _custom_jvp_jvp_wrap(jvp, self.tag, in_treedef)
		out_leaves = primitive.bind_with_trace(
			self.parent_trace,
			(fun, jvp, *in_leaves),
			dict(symbolic_zeros=symbolic_zeros),
		)
		_, out_treedef = lu.merge_linear_aux(out_treedef1, out_treedef2)
		out_values = tu.tree_unflatten(out_treedef, out_leaves)
		return [_CustomTracer(self, x) for x in out_values]

	# TODO: add other process_* rules


@lu.transformation_with_aux
def _custom_jvp_fun_wrap(tag, in_treedef, *in_leaves):
	in_values = tu.tree_unflatten(in_treedef, in_leaves)
	with core.take_current_trace() as parent_trace:
		trace = _Tracer(parent_trace, tag)
		in_tracers = [x if type(x) is SZ else _CustomTracer(trace, x) for x in in_values]
		with core.set_current_trace(trace):
			out_tracers = yield in_tracers, {}
			out_tracers = [
				jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t
				for t in out_tracers
			]
			out_values = [trace.to_value(t) for t in out_tracers]
			del out_tracers
		del trace, in_tracers
	out_leaves, out_treedef = tu.tree_flatten(out_values)
	yield out_leaves, out_treedef


@lu.transformation_with_aux
def _custom_jvp_jvp_wrap(tag, in_treedef, *in_primals_and_tangents):
	in_primals = in_primals_and_tangents[: len(in_primals_and_tangents) // 2]
	in_tangents = in_primals_and_tangents[len(in_primals_and_tangents) // 2 :]
	in_primal_values = tu.tree_unflatten(in_treedef, in_primals)
	in_tangent_values = tu.tree_unflatten(in_treedef, in_tangents)
	with core.take_current_trace() as parent_trace:
		trace = _Tracer(parent_trace, tag)
		in_tracers = [
			_CustomTracer(trace, x) for x in it.chain(in_primal_values, in_tangent_values)
		]
		with core.set_current_trace(trace):
			out_tracers = yield in_tracers, {}
			out_tracers = [
				jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t
				for t in out_tracers
			]
			out_values = [trace.to_value(t) for t in out_tracers]
			out_primal_values = out_values[: len(out_values) // 2]
			out_tangent_values = out_values[len(out_values) // 2 :]
			out_primal_values2 = []
			out_tangent_values2 = []
			assert len(out_primal_values) == len(out_tangent_values)
			for primal, tangent in zip(out_primal_values, out_tangent_values):  # noqa
				if primal.__class__ != tangent.__class__:
					primal = primal.materialize()
					tangent = tangent.materialize()
				out_primal_values2.append(primal)
				out_tangent_values2.append(tangent)
			del out_tracers
		del trace, in_tracers
	out_primals, out_primal_treedef = tu.tree_flatten(out_primal_values2)
	out_tangents, out_tangent_treedef = tu.tree_flatten(out_tangent_values2)
	if out_primal_treedef != out_tangent_treedef:
		raise ValueError(
			"Primals and tangents had the same class, but different flattened results."
		)
	yield out_primals + out_tangents, out_primal_treedef


def _wrap_tracer(trace: _Tracer, x):
	if _is_value(x):
		return _CustomTracer(trace, x)
	else:
		return x


def _unwrap_tracer(trace, x):
	if is_array(x):
		x = trace.full_raise(x)
	if isinstance(x, _CustomTracer):
		if isinstance(x.value, _DenseArrayValue):
			return x.value.array
		else:
			return x.value
	else:
		return x


class _Transform(PyTree, tp.Generic[CT]):
	fn: CT
	filter_spec: tp.Dict[str, tp.Union[bool, Callable[[tp.Any], bool]]]
	dynamic: bool = field(static=True)

	@property
	def __wrapped__(self) -> CT:
		return self.fn

	def __call__(self, *args, **kwargs):
		dynamic, static = partition(
			(self.fn, args, kwargs),
			self.filter_spec,
			is_leaf=_is_value,
		)
		tag = core.TraceTag()
		with core.take_current_trace() as parent_trace:
			trace = _Tracer(parent_trace, tag)
			dynamic = tu.tree_map(
				ft.partial(_wrap_tracer, trace),
				dynamic,
				is_leaf=_is_value,
			)
			fn, args, kwargs = combine(dynamic, static)
			with core.set_current_trace(trace):
				out = fn(*args, **kwargs)
			out = tu.tree_map(ft.partial(_unwrap_tracer, trace), out)
			return out

	def __get__(self, instance: tp.Union[object, None], owner: tp.Any):
		if instance is None:
			return self
		return Partial(self, instance)


def implicit(
	fn: CT,
	filter_spec: tp.Dict[str, tp.Union[bool, Callable[[tp.Any], bool]]] = True,
) -> _Transform[CT]:
	return tp.cast(
		_Transform[CT],
		module_update_wrapper(_Transform(fn, filter_spec, dynamic=False)),
	)


class Value(PyTree):
	@staticmethod
	def default(
		primitive: core.Primitive,
		values: Sequence[tp.Union[chex.Array, "Value"]],
		params,
	) -> tp.Union[chex.Array, "Value", Sequence[tp.Union[chex.Array, "Value"]]]:
		arrays: list[chex.Array] = []

		for x in values:
			if _is_value(x):
				arrays.append(x.materialize())
			elif is_array(x):
				arrays.append(tp.cast(chex.Array, x))
			else:
				arrays.append(x)

		subfuns, bind_params = primitive.get_bind_params(params)
		return primitive.bind(*subfuns, *arrays, **bind_params)

	@abc.abstractmethod
	def aval(self) -> core.AbstractValue: ...
	@abc.abstractmethod
	def materialize(self) -> tp.Any: ...


def _is_value(x) -> TypeGuard[Value]:
	return isinstance(x, Value)


class ArrayValue(Value):
	@abc.abstractmethod
	def materialize(self) -> chex.Array:
		pass

	def aval(self) -> core.ShapedArray:
		return jax.core.get_aval(jax.eval_shape(lambda: self.materialize()))

	@property
	def shape(self):
		return self.aval().shape

	@property
	def dtype(self):
		return self.aval().dtype

	@property
	def ndim(self):
		return self.aval().ndim

	@property
	def size(self):
		return self.aval().size


class _DenseArrayValue(ArrayValue):
	array: chex.Array

	def materialize(self) -> chex.Array:
		return self.array

	def aval(self) -> core.ShapedArray:
		return core.get_aval(self.array)


@register(jax._src.pjit.pjit_p)
def _(*args: tp.Union[ArrayValue, Value], jaxpr, inline, **kwargs):
	del kwargs
	fun = implicit(core.jaxpr_as_fun(jaxpr))
	if inline:
		return fun(*args)
	else:
		leaves, treedef = tu.tree_flatten(args)
		flat_fun = lambda x: fun(*tu.tree_unflatten(treedef, x))  # noqa
		return jax.jit(flat_fun)(leaves)


@register("while")
def _(
	*args: tp.Any,
	cond_nconsts: int,
	cond_jaxpr,
	body_nconsts: int,
	body_jaxpr,
):
	cond_consts = args[:cond_nconsts]
	body_consts = args[cond_nconsts : cond_nconsts + body_nconsts]
	init_vals = args[cond_nconsts + body_nconsts :]

	# compute jaxpr of ified body and condition function
	_cond_fn = implicit(core.jaxpr_as_fun(cond_jaxpr))
	_cond_jaxpr = jax.make_jaxpr(_cond_fn)(*cond_consts, *init_vals)
	_body_fn = implicit(core.jaxpr_as_fun(body_jaxpr))
	_body_jaxpr = jax.make_jaxpr(_body_fn)(*body_consts, *init_vals)

	cond_leaves, _ = tu.tree_flatten(cond_consts)
	body_leaves, _ = tu.tree_flatten(body_consts)
	init_val_leaves, val_treedef = tu.tree_flatten(init_vals)

	out_val = jax.lax.while_p.bind(
		*cond_leaves,
		*body_leaves,
		*init_val_leaves,
		cond_nconsts=cond_nconsts,
		cond_jaxpr=_cond_jaxpr,
		body_nconsts=body_nconsts,
		body_jaxpr=_body_jaxpr,
	)
	result = tu.tree_unflatten(val_treedef, out_val)
	return result


_sentinel = object()


@register("cond")
def _(
	index: chex.Array,
	*args: tp.Any,
	branches: tuple,
	linear=_sentinel,
):
	flat_args, in_tree = tu.tree_flatten(args)

	out_trees = []
	_branches = []
	for jaxpr in branches:

		def flat__call(flat_args):
			args = tu.tree_unflatten(in_tree, flat_args)
			out = implicit(core.jaxpr_as_fun(jaxpr))(*args)  # noqa
			flat_out, out_tree = tu.tree_flatten(out)
			out_trees.append(out_tree)
			return flat_out

		_jaxpr = jax.make_jaxpr(flat__call)(flat_args)
		_branches.append(_jaxpr)

	if tp.Any(tree_outs_i != out_trees[0] for tree_outs_i in out_trees[1:]):
		raise TypeError("all branches output must have the same pytree.")

	if linear is _sentinel:
		maybe_linear = {}
	else:
		maybe_linear = dict(linear=linear)
	out_val = jax.lax.cond_p.bind(
		index, *flat_args, branches=tuple(_branches), **maybe_linear
	)
	result = tu.tree_unflatten(out_trees[0], out_val)
	return result
