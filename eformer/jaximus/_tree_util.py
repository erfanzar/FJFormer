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

import functools
import typing as tp

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax import tree_util as tu

T = tp.TypeVar("T")

if tp.TYPE_CHECKING:
	from ._core import PyTree as PyTreeType
else:
	PyTreeType = tp.Any

FilterSpec = tp.Union[bool, tp.Callable[[tp.Any], bool]]
IsLeafFn = tp.Callable[[tp.Any], bool]
TreeDict = tp.Dict[str, bool]


def _array_equal(x, y, npi, rtol, atol):
	assert x.dtype == y.dtype
	if (
		isinstance(rtol, (int, float))
		and isinstance(atol, (int, float))
		and rtol == 0
		and atol == 0
	) or not npi.issubdtype(x.dtype, npi.inexact):
		return npi.all(x == y)
	else:
		return npi.allclose(x, y, rtol=rtol, atol=atol)


def is_array(element: tp.Any) -> bool:
	"""Returns `True` if `element` is a JAX array or NumPy array."""
	return isinstance(element, (np.ndarray, np.generic, Array))


def is_array_like(element: tp.Any) -> bool:
	return isinstance(
		element,
		(
			Array,
			np.ndarray,
			np.generic,
			float,
			complex,
			bool,
			int,
		),
	) or hasattr(element, "__jax_array__")


class TreeFilter(tp.Protocol):
	"""tp.Protocol for tree filter functions."""

	def __call__(self, mask: tp.Any, arg: tp.Any) -> TreeDict: ...


def partition(
	pytree: PyTreeType,
	filter_spec: FilterSpec,
	replace: tp.Any = None,
	is_leaf: tp.Optional[IsLeafFn] = None,
) -> tp.Tuple[PyTreeType, PyTreeType]:
	def _make_filter_tree(il):
		def _filter_tree(mask: FilterSpec, arg: tp.Any) -> TreeDict:
			if isinstance(mask, bool):
				return tu.tree_map(lambda _: mask, arg, is_leaf=il)
			elif callable(mask):
				return tu.tree_map(mask, arg, is_leaf=il)
			else:
				raise ValueError(f"filter_spec must be bool or callable, got {type(mask)}")

		return _filter_tree

	filter_tree = tu.tree_map(_make_filter_tree(is_leaf), filter_spec, pytree)
	return (
		tu.tree_map(lambda mask, x: x if mask else replace, filter_tree, pytree),
		tu.tree_map(lambda mask, x: replace if mask else x, filter_tree, pytree),
	)


def combine(*pytrees: PyTreeType, is_leaf: tp.Optional[IsLeafFn] = None) -> PyTreeType:
	"""
	Combines multiple PyTrees into a single PyTreeType.

	Args:
	    *pytrees: PyTrees to combine
	    is_leaf: tp.Optional function to determine if a node is a leaf

	Returns:
	    Combined PyTreeType
	"""

	def _combine(*args: tp.Any) -> tp.Any:
		"""Returns first non-None value from args."""
		return next((arg for arg in args if arg is not None), None)

	def _is_none(x: tp.Any) -> bool:
		"""Checks if value is None."""
		return x is None

	if is_leaf is None:
		_is_leaf = _is_none
	else:

		def _is_leaf(x: tp.Any) -> bool:
			return _is_none(x) or is_leaf(x)

	return tu.tree_map(_combine, *pytrees, is_leaf=_is_leaf)


def tree_equal(
	*pytrees: PyTreeType,
	typematch: bool = False,
	rtol=0.0,
	atol=0.0,
) -> bool:
	flat, treedef = tu.tree_flatten(pytrees[0])
	traced_out = True
	for pytree in pytrees[1:]:
		flat_, treedef_ = tu.tree_flatten(pytree)
		if treedef_ != treedef:
			return False
		assert len(flat) == len(flat_)
		for elem, elem_ in zip(flat, flat_):  # noqa
			if typematch:
				if type(elem) != type(elem_):  # noqa
					return False
			if isinstance(elem, (np.ndarray, np.generic)) and isinstance(
				elem_, (np.ndarray, np.generic)
			):
				if (
					(elem.shape != elem_.shape)
					or (elem.dtype != elem_.dtype)
					or not _array_equal(elem, elem_, np, rtol, atol)
				):
					return False
			elif is_array(elem):
				if is_array(elem_):
					if (elem.shape != elem_.shape) or (elem.dtype != elem_.dtype):
						return False
					traced_out = traced_out & _array_equal(elem, elem_, jax.numpy, rtol, atol)
				else:
					return False
			else:
				if is_array(elem_):
					return False
				else:
					if elem != elem_:
						return False
	return traced_out


def tree_map_with_path(
	f: tp.Callable,
	tree: PyTreeType,
	is_leaf: tp.Optional[IsLeafFn] = None,
) -> PyTreeType:
	"""Maps a function over a pytree while providing the path to each leaf.

	Args:
	    f: Function that takes (path, leaf_value) as arguments
	    tree: Input pytree
	    is_leaf: Optional function to determine if a node is a leaf

	Returns:
	    PyTreeType with mapped values
	"""

	def _walk(path: tp.Tuple[str, ...], x):
		if is_leaf is not None and is_leaf(x):
			return f(path, x)
		elif isinstance(x, (list, tuple)):
			return type(x)([_walk(path + (str(i),), v) for i, v in enumerate(x)])
		elif isinstance(x, dict):
			return {k: _walk(path + (str(k),), v) for k, v in x.items()}
		else:
			return f(path, x)

	return _walk((), tree)


def tree_flatten_with_paths(
	tree: PyTreeType,
	is_leaf: tp.Optional[IsLeafFn] = None,
) -> tp.Tuple[tp.List[tp.Tuple[tp.Tuple, tp.Any]], tu.PyTreeDef]:
	"""Flattens a pytree while keeping track of paths to leaves.

	Args:
	    tree: Input pytree
	    is_leaf: Optional function to determine if a node is a leaf

	Returns:
	    Tuple of (list of (path, value) pairs, treedef)
	"""
	paths_and_vals = []

	def _record_path(path, x):
		paths_and_vals.append((path, x))
		return x

	tree_map_with_path(_record_path, tree, is_leaf=is_leaf)
	treedef = tu.tree_structure(tree)
	return paths_and_vals, treedef


def tree_leaves_with_paths(
	tree: PyTreeType, is_leaf: tp.Optional[IsLeafFn] = None
) -> tp.List[tp.Tuple[tp.Tuple, tp.Any]]:
	"""Returns list of (path, leaf_value) pairs in the pytree."""
	paths_and_vals, _ = tree_flatten_with_paths(tree, is_leaf=is_leaf)
	return paths_and_vals


def tree_structure_equal(tree1: PyTreeType, tree2: PyTreeType) -> bool:
	"""Returns True if two pytrees have the same structure."""
	try:
		return tu.tree_structure(tree1) == tu.tree_structure(tree2)
	except Exception:
		return False


def tree_filter(tree: PyTreeType, predicate: tp.Callable[[tp.Any], bool]) -> PyTreeType:
	"""Filters a pytree keeping only leaves that satisfy the predicate."""
	flat, treedef = tu.tree_flatten(tree)
	filtered = [x for x in flat if predicate(x)]
	return tu.tree_unflatten(treedef, filtered)


def tree_reduce(
	f: tp.Callable[[tp.Any, tp.Any], tp.Any],
	tree: PyTreeType,
	initial_value: tp.Any,
) -> tp.Any:
	"""Reduces a pytree to a single value using the given function."""
	flat, _ = tu.tree_flatten(tree)
	return functools.reduce(f, flat, initial_value)


def tree_shape(tree: PyTreeType) -> PyTreeType:
	"""Returns a pytree with the same structure but with shapes instead of arrays."""
	return tu.tree_map(lambda x: x.shape if hasattr(x, "shape") else None, tree)


def tree_dtype(tree: PyTreeType) -> PyTreeType:
	"""Returns a pytree with the same structure but with dtypes instead of arrays."""
	return tu.tree_map(lambda x: x.dtype if hasattr(x, "dtype") else type(x), tree)


def tree_size(tree: PyTreeType) -> int:
	"""Returns the total number of elements in all arrays in the pytree."""
	return sum(x.size if hasattr(x, "size") else 1 for x in tu.tree_leaves(tree))


def tree_concatenate(trees: tp.List[PyTreeType], axis: int = 0) -> PyTreeType:
	"""Concatenates corresponding arrays in a list of pytrees."""
	return tu.tree_map(lambda *xs: jnp.concatenate(xs, axis=axis), *trees)


def tree_stack(trees: tp.List[PyTreeType], axis: int = 0) -> PyTreeType:
	"""Stacks corresponding arrays in a list of pytrees."""
	return tu.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *trees)


def tree_where(condition: PyTreeType, x: PyTreeType, y: PyTreeType) -> PyTreeType:
	"""Element-wise where operation on pytrees."""
	return tu.tree_map(lambda c, a, b: jnp.where(c, a, b), condition, x, y)


def tree_zeros_like(tree: PyTreeType) -> PyTreeType:
	"""Creates a pytree of zeros with the same structure and shapes."""
	return tu.tree_map(lambda x: jnp.zeros_like(x) if is_array_like(x) else x, tree)


def tree_ones_like(tree: PyTreeType) -> PyTreeType:
	"""Creates a pytree of ones with the same structure and shapes."""
	return tu.tree_map(lambda x: jnp.ones_like(x) if is_array_like(x) else x, tree)


def tree_to_device(tree: PyTreeType, device: jax.Device) -> PyTreeType:
	"""Moves all arrays in a pytree to the specified device."""
	return tu.tree_map(
		lambda x: jax.device_put(x, device) if is_array_like(x) else x, tree
	)


class TreeDiff:
	"""Utility class for comparing pytrees and finding differences."""

	@staticmethod
	def compare(tree1: PyTreeType, tree2: PyTreeType) -> tp.Dict[str, tp.List[str]]:
		"""Compares two pytrees and returns differences in structure, shape, and values."""
		differences = {"structure": [], "shape": [], "value": []}

		def _compare_leaves(path: tp.Tuple[str, ...], x1: tp.Any, x2: tp.Any):
			path_str = "/".join(map(str, path))
			if type(x1) != type(x2):  # noqa
				differences["structure"].append(
					f"{path_str}: type mismatch ({type(x1)} vs {type(x2)})"
				)
			elif is_array_like(x1) and is_array_like(x2):
				if x1.shape != x2.shape:
					differences["shape"].append(
						f"{path_str}: shape mismatch ({x1.shape} vs {x2.shape})"
					)
				elif not jnp.allclose(x1, x2):
					differences["value"].append(f"{path_str}: value mismatch")
			elif x1 != x2:
				differences["value"].append(f"{path_str}: value mismatch ({x1} vs {x2})")

		try:
			tu.tree_map_with_path(_compare_leaves, tree1, tree2)
		except Exception:
			differences["structure"].append("Trees have different structures")

		return {k: v for k, v in differences.items() if v}


def tree_summary(tree: PyTreeType) -> tp.Dict[str, tp.Any]:
	"""Generates a summary of the pytree structure and contents."""
	summary = {
		"num_leaves": len(tu.tree_leaves(tree)),
		"total_parameters": tree_size(tree),
		"shapes": tree_shape(tree),
		"dtypes": tree_dtype(tree),
		"memory_usage": sum(
			x.size * x.dtype.itemsize
			for x in tu.tree_leaves(tree)
			if hasattr(x, "size") and hasattr(x, "dtype")
		),
	}
	return summary
