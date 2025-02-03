import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eformer.jaximus import PyTree
from eformer.jaximus._tree_util import *


class DummyDataclass(PyTree):
	x: jnp.ndarray
	y: int


class DummyNamedTuple(NamedTuple):
	a: jnp.ndarray
	b: str


@pytest.fixture
def sample_trees():
	"""Fixture providing various sample trees for testing."""
	simple_tree = {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}
	nested_tree = {
		"x": {"y": jnp.array([1, 2])},
		"z": [jnp.array([3, 4]), jnp.array([5, 6])],
	}
	mixed_tree = {
		"array": jnp.array([1, 2]),
		"number": 42,
		"string": "hello",
		"dataclass": DummyDataclass(jnp.array([1, 2]), 5),
		"namedtuple": DummyNamedTuple(jnp.array([3, 4]), "test"),
	}
	return simple_tree, nested_tree, mixed_tree


def test_is_array():
	"""Test is_array function."""
	assert is_array(jnp.array([1, 2]))
	assert is_array(np.array([1, 2]))
	assert not is_array([1, 2])
	assert not is_array(42)
	assert not is_array("string")


def test_is_array_like():
	"""Test is_array_like function."""
	assert is_array_like(jnp.array([1, 2]))
	assert is_array_like(np.array([1, 2]))
	assert is_array_like(42)
	assert is_array_like(3.14)
	assert not is_array_like("string")


def test_partition():
	"""Test partition function."""
	tree = {"a": jnp.array([1]), "b": jnp.array([2])}

	# Test with boolean filter
	true_part, false_part = partition(tree, True)
	assert tree_equal(true_part, tree)
	assert tree_equal(false_part, {"a": None, "b": None})

	# Test with callable filter
	filter_fn = lambda x: isinstance(x, jax.Array)
	array_part, non_array_part = partition(tree, filter_fn)
	assert tree_equal(array_part, tree)


def test_combine(sample_trees):
	"""Test combine function."""
	tree1 = {"a": jnp.array([1]), "b": None}
	tree2 = {"a": None, "b": jnp.array([2])}
	combined = combine(tree1, tree2)
	assert tree_equal(combined, {"a": jnp.array([1]), "b": jnp.array([2])})


def test_tree_map_with_path(sample_trees):
	"""Test tree_map_with_path function."""
	simple_tree, _, _ = sample_trees

	def path_recorder(path, value):
		return f"{'/'.join(map(str, path))}:{value}"

	mapped = tree_map_with_path(path_recorder, simple_tree)
	assert mapped["a"].startswith("a:")
	assert mapped["b"].startswith("b:")


def test_tree_flatten_with_paths(sample_trees):
	"""Test tree_flatten_with_paths function."""
	_, nested_tree, _ = sample_trees

	paths_and_vals, treedef = tree_flatten_with_paths(nested_tree)
	assert len(paths_and_vals) == 3  # Should have 3 arrays
	assert all(isinstance(path, tuple) for path, _ in paths_and_vals)


def test_tree_structure_equal(sample_trees):
	"""Test tree_structure_equal function."""
	simple_tree, _, _ = sample_trees

	same_structure = {"a": jnp.array([5, 6]), "b": jnp.array([7, 8])}
	different_structure = {"a": jnp.array([1, 2]), "c": jnp.array([3, 4])}

	assert tree_structure_equal(simple_tree, same_structure)
	assert not tree_structure_equal(simple_tree, different_structure)


def test_tree_reduce(sample_trees):
	"""Test tree_reduce function."""
	simple_tree, _, _ = sample_trees

	# Sum all array elements
	total = tree_reduce(
		lambda x, y: x + y.sum() if isinstance(y, jax.Array) else x, simple_tree, 0.0
	)
	expected_sum = sum(x.sum() for x in simple_tree.values())
	assert jnp.allclose(total, expected_sum)


def test_tree_shape(sample_trees):
	"""Test tree_shape function."""
	simple_tree, _, _ = sample_trees
	shapes = tree_shape(simple_tree)
	assert all(shape == (2,) for shape in shapes.values())


def test_tree_dtype(sample_trees):
	"""Test tree_dtype function."""
	simple_tree, _, _ = sample_trees
	dtypes = tree_dtype(simple_tree)
	assert all(dtype == jnp.int32 for dtype in dtypes.values())


def test_tree_size(sample_trees):
	"""Test tree_size function."""
	simple_tree, _, _ = sample_trees
	assert tree_size(simple_tree) == 4  # Two arrays of size 2 each


def test_tree_concatenate():
	"""Test tree_concatenate function."""
	tree1 = {"a": jnp.array([1, 2])}
	tree2 = {"a": jnp.array([3, 4])}
	result = tree_concatenate([tree1, tree2])
	assert jnp.array_equal(result["a"], jnp.array([1, 2, 3, 4]))


def test_tree_stack():
	"""Test tree_stack function."""
	tree1 = {"a": jnp.array([1, 2])}
	tree2 = {"a": jnp.array([3, 4])}
	result = tree_stack([tree1, tree2])
	assert result["a"].shape == (2, 2)


def test_tree_where():
	"""Test tree_where function."""
	condition = {"a": jnp.array([True, False])}
	x = {"a": jnp.array([1, 2])}
	y = {"a": jnp.array([3, 4])}
	result = tree_where(condition, x, y)
	assert jnp.array_equal(result["a"], jnp.array([1, 4]))


def test_tree_zeros_like(sample_trees):
	"""Test tree_zeros_like function."""
	simple_tree, _, _ = sample_trees
	zeros = tree_zeros_like(simple_tree)
	assert all(jnp.all(x == 0) for x in zeros.values())


def test_tree_ones_like(sample_trees):
	"""Test tree_ones_like function."""
	simple_tree, _, _ = sample_trees
	ones = tree_ones_like(simple_tree)
	assert all(jnp.all(x == 1) for x in ones.values())


def test_tree_to_device(sample_trees):
	"""Test tree_to_device function."""
	simple_tree, _, _ = sample_trees
	device = jax.devices()[0]
	tree_to_device(simple_tree, device)


def test_tree_diff(sample_trees):
	"""Test TreeDiff class."""
	simple_tree, _, _ = sample_trees
	different_tree = {
		"a": jnp.array([1, 3]),  # Different values
		"b": jnp.array([3, 4, 5]),  # Different shape
	}

	differences = TreeDiff.compare(simple_tree, different_tree)
	assert "shape" in differences
	assert "value" in differences


def test_tree_summary(sample_trees):
	"""Test tree_summary function."""
	simple_tree, _, _ = sample_trees
	summary = tree_summary(simple_tree)

	assert summary["num_leaves"] == 2
	assert summary["total_parameters"] == 4
	assert "shapes" in summary
	assert "dtypes" in summary
	assert "memory_usage" in summary


@pytest.mark.parametrize(
	"tree,expected_size",
	[
		({"a": jnp.zeros(5)}, 5),
		({"a": jnp.zeros((2, 3))}, 6),
		({"a": jnp.zeros(2), "b": jnp.zeros(3)}, 5),
	],
)
def test_tree_size_parametrized(tree, expected_size):
	"""Parametrized tests for tree_size with different inputs."""
	assert tree_size(tree) == expected_size


if __name__ == "__main__":
	pytest.main([__file__])
