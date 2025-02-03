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


"""
A module for pretty-printing PyTree structures with customizable formatting options.
"""

import dataclasses
import sys
from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeVar, Union

import jax
import jax._src.pretty_printer as pp
import numpy as np

T = TypeVar("T")
Dataclass = Any
NamedTuple = Any


@dataclasses.dataclass
class FormattingOptions:
	"""Configuration options for tree printing."""

	indent: int = 2
	width: int = 80
	short_arrays: bool = True
	struct_as_array: bool = False
	follow_wrapped: bool = True
	truncate_leaf: Callable[[Any], bool] = lambda _: False


def bracketed(
	name: Optional[pp.Doc],
	indent: int,
	objs: Sequence[pp.Doc],
	lbracket: str,
	rbracket: str,
) -> pp.Doc:
	"""Creates a bracketed document with optional name and indentation."""
	nested = pp.concat(
		[
			pp.nest(
				indent,
				pp.concat([pp.brk(""), pp.join(pp.concat([pp.text(","), pp.brk()]), objs)]),
			),
			pp.brk(""),
		]
	)
	elements = [pp.text(lbracket), nested, pp.text(rbracket)]
	if name is not None:
		elements.insert(0, name)
	return pp.group(pp.concat(elements))


class TreeFormatter:
	"""Handles the formatting of different types in the PyTree."""

	@staticmethod
	def format_named_objects(pairs: list[tuple[str, Any]], **kwargs) -> list[pp.Doc]:
		"""Formats name-value pairs."""
		return [
			pp.concat([pp.text(f"{key}="), tree_pp(value, **kwargs)]) for key, value in pairs
		]

	@staticmethod
	def format_dataclass(obj: Dataclass, **kwargs) -> pp.Doc:
		"""Formats a dataclass object."""
		objs = TreeFormatter.format_named_objects(
			[
				(field.name, getattr(obj, field.name, "<None>"))
				for field in dataclasses.fields(obj)
				if field.repr
			],
			**kwargs,
		)
		return bracketed(
			name=pp.text(obj.__class__.__name__),
			indent=kwargs["indent"],
			objs=objs,
			lbracket="(",
			rbracket=")",
		)

	@staticmethod
	def format_namedtuple(obj: NamedTuple, **kwargs) -> pp.Doc:
		"""Formats a named tuple object."""
		objs = TreeFormatter.format_named_objects(
			[(name, getattr(obj, name)) for name in obj._fields], **kwargs
		)
		return bracketed(
			name=pp.text(obj.__class__.__name__),
			indent=kwargs["indent"],
			objs=objs,
			lbracket="(",
			rbracket=")",
		)

	@staticmethod
	def format_list(obj: list, **kwargs) -> pp.Doc:
		"""Formats a list object."""
		return bracketed(
			name=None,
			indent=kwargs["indent"],
			objs=[tree_pp(x, **kwargs) for x in obj],
			lbracket="[",
			rbracket="]",
		)

	@staticmethod
	def format_tuple(obj: tuple, **kwargs) -> pp.Doc:
		"""Formats a tuple object."""
		objs = (
			[pp.concat([tree_pp(obj[0], **kwargs), pp.text(",")])]
			if len(obj) == 1
			else [tree_pp(x, **kwargs) for x in obj]
		)
		return bracketed(
			name=None, indent=kwargs["indent"], objs=objs, lbracket="(", rbracket=")"
		)

	@staticmethod
	def format_dict(obj: dict, **kwargs) -> pp.Doc:
		"""Formats a dictionary object."""

		def format_entry(key: Any, value: Any) -> pp.Doc:
			return pp.concat(
				[tree_pp(key, **kwargs), pp.text(":"), pp.brk(), tree_pp(value, **kwargs)]
			)

		objs = [format_entry(key, value) for key, value in obj.items()]
		return bracketed(
			name=None, indent=kwargs["indent"], objs=objs, lbracket="{", rbracket="}"
		)


class ArrayFormatter:
	"""Handles the formatting of array-like objects."""

	@staticmethod
	def format_short_array(
		shape: tuple[int, ...], dtype: str, kind: Optional[str] = None
	) -> pp.Doc:
		"""Creates a short representation of an array."""
		dtype_abbrev = (
			dtype.replace("float", "f")
			.replace("uint", "u")
			.replace("int", "i")
			.replace("complex", "c")
		)
		shape_str = ",".join(map(str, shape))
		text = f"{dtype_abbrev}[{shape_str}]"
		if kind is not None:
			text = f"{text}({kind})"
		return pp.text(text)

	@staticmethod
	def format_array(
		obj: Union[jax.Array, jax.ShapeDtypeStruct, np.ndarray],
		**kwargs,
	) -> pp.Doc:
		"""Formats an array-like object."""
		if not kwargs.get("short_arrays", True):
			return pp.text(repr(obj))

		if "torch" in sys.modules and isinstance(obj, sys.modules["torch"].Tensor):
			dtype = repr(obj.dtype).split(".")[1]
			kind = "torch"
		else:
			dtype = obj.dtype.name
			if isinstance(obj, (jax.Array, jax.ShapeDtypeStruct)):
				if getattr(obj, "weak_type", False):
					dtype = f"weak_{dtype}"
				kind = None
			elif isinstance(obj, np.ndarray):
				kind = "numpy"
			else:
				kind = "unknown"

		return ArrayFormatter.format_short_array(obj.shape, dtype, kind)


def tree_pp(obj: Any, **kwargs) -> pp.Doc:
	"""
	Pretty prints a PyTree structure.

	Args:
	    obj: The object to format
	    **kwargs: Formatting options

	Returns:
	    A pretty-printed document
	"""
	if kwargs["truncate_leaf"](obj):
		return pp.text(f"{type(obj).__name__}(...)")

	if hasattr(obj, "__tree_pp__"):
		custom_pp = obj.__tree_pp__(**kwargs)
		if custom_pp is not NotImplemented:
			return pp.group(custom_pp)

	# Handle different types
	if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
		return TreeFormatter.format_dataclass(obj, **kwargs)
	elif isinstance(obj, list):
		return TreeFormatter.format_list(obj, **kwargs)
	elif isinstance(obj, dict):
		return TreeFormatter.format_dict(obj, **kwargs)
	elif isinstance(obj, tuple):
		return (
			TreeFormatter.format_namedtuple(obj, **kwargs)
			if hasattr(obj, "_fields")
			else TreeFormatter.format_tuple(obj, **kwargs)
		)
	elif (
		isinstance(obj, (np.ndarray, jax.Array))
		or ("torch" in sys.modules and isinstance(obj, sys.modules["torch"].Tensor))
		or (kwargs.get("struct_as_array", False) and isinstance(obj, jax.ShapeDtypeStruct))
	):
		return ArrayFormatter.format_array(obj, **kwargs)

	# Handle special cases
	if isinstance(obj, (jax.custom_jvp, jax.custom_vjp)):
		return tree_pp(obj.__wrapped__, **kwargs)
	elif hasattr(obj, "__wrapped__") and kwargs.get("follow_wrapped", True):
		kwargs["wrapped"] = True
		return tree_pp(obj.__wrapped__, **kwargs)

	return pp.text(repr(obj))


def tree_pformat(
	pytree: Any,
	options: Optional[FormattingOptions] = None,
) -> str:
	"""
	Format a PyTree structure as a string.

	Args:
	    pytree: The structure to format
	    options: Formatting options

	Returns:
	    A formatted string representation
	"""
	if options is None:
		options = FormattingOptions()

	return tree_pp(
		pytree,
		indent=options.indent,
		short_arrays=options.short_arrays,
		struct_as_array=options.struct_as_array,
		follow_wrapped=options.follow_wrapped,
		truncate_leaf=options.truncate_leaf,
	).format(width=options.width)
