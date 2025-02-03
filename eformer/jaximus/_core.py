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


from __future__ import annotations

import abc
import dataclasses
import functools as ft
import inspect
import types
import typing as tp
import warnings
import weakref

import jax.tree_util as tu
import numpy as np
from jax import Array
from typing_extensions import ParamSpec, dataclass_transform

from ._pprint import tree_pformat
from ._tree_util import tree_equal

_T = tp.TypeVar("_T")
_P = ParamSpec("_P")

_is_force_abstract = weakref.WeakKeyDictionary()
_is_strict = weakref.WeakKeyDictionary()
_has_dataclass_init = weakref.WeakKeyDictionary()

if tp.TYPE_CHECKING:
	AbstractVar: tp.TypeAlias = tp.Annotated[_T, "AbstractVar"]
	from typing import ClassVar as AbstractClassVar
else:

	class AbstractVar(tp.Generic[_T]): ...

	class AbstractClassVar(tp.Generic[_T]): ...


class _Initable:
	def __init_subclass__(cls, **kwargs):
		del kwargs


class WithRepr(type):
	def __new__(mcs, obj, string):
		out = super().__new__(mcs, string, (), {})
		out.__module__ = "builtins"
		return out

	def __init__(cls, obj, string):
		cls.obj = obj
		cls.string = string

	def __repr__(cls):
		return cls.string

	def __call__(cls, *args, **kwargs):
		return cls.obj(*args, **kwargs)


def doc_repr(obj: _T, string: str) -> _T:
	if tp.TYPE_CHECKING:
		return obj
	else:
		if getattr(tp, "GENERATING_DOCUMENTATION", False):
			return WithRepr(obj, string)
		else:
			return obj


_dummy_abstract = abc.abstractmethod(lambda self: 1)
_converter_sentinel: tp.Any = doc_repr(object(), "lambda x: x")


def field(
	*,
	converter: tp.Callable[[tp.Any], tp.Any] = _converter_sentinel,
	static: bool = False,
	**kwargs,
):
	try:
		metadata = dict(kwargs.pop("metadata"))
	except KeyError:
		metadata = {}
	if "converter" in metadata:
		raise ValueError("Cannot use metadata with `static` already set.")
	if "static" in metadata:
		raise ValueError("Cannot use metadata with `static` already set.")
	if converter is not _converter_sentinel:
		metadata["converter"] = converter
	if static:
		metadata["static"] = True
	return dataclasses.field(metadata=metadata, **kwargs)


def static_field(**kwargs):
	return field(**kwargs, static=True)


def _is_abstract(cls):
	return (
		_is_force_abstract[cls]
		or len(cls.__abstractmethods__) > 0
		or len(cls.__abstractvars__) > 0
		or len(cls.__abstractclassvars__) > 0
	)


_wrapper_field_names = {
	"__module__",
	"__name__",
	"__qualname__",
	"__doc__",
	"__annotations__",
}


@dataclass_transform()
def dataclass(**kwargs):
	def make_dataclass(cls):
		try:
			annotations = cls.__dict__["__annotations__"]
		except KeyError:
			cls = dataclasses.dataclass(**kwargs)(cls)
		else:
			new_annotations = dict(annotations)
			for name, annotation in annotations.items():
				is_abstract, _ = _process_annotation(annotation)
				if is_abstract:
					new_annotations.pop(name)
			cls.__annotations__ = new_annotations
			cls = dataclasses.dataclass(**kwargs)(cls)
			cls.__annotations__ = annotations
		return cls

	return make_dataclass


def _convert_fields(module, init: bool):
	for field in dataclasses.fields(module):
		if field.init is init:
			try:
				converter = field.metadata["converter"]
			except KeyError:
				pass
			else:
				try:
					value = getattr(module, field.name)
				except AttributeError:
					pass
				else:
					setattr(module, field.name, converter(value))


class _wrap_method:
	def __init__(self, method):
		self.method = method
		if getattr(self.method, "__isabstractmethod__", False):
			self.__isabstractmethod__ = self.method.__isabstractmethod__

	def __get__(self, instance, owner):
		if instance is None:
			return self.method
		else:
			return _module_update_wrapper(
				BoundMethod(self.method, instance),
				None,
				inplace=True,
			)


def _not_magic(k: str) -> bool:
	return not (k.startswith("__") and k.endswith("__"))


def _is_special_form(cls):
	if cls is _Initable:
		return True
	if cls.__module__ in ("typing", "typing_extensions", "collections.abc"):
		return True
	if tp.Protocol in cls.__bases__:
		return True
	return False


@ft.lru_cache(maxsize=128)
def _make_initable(
	cls: _ActualModuleMeta,
	init,
	post_init,
	wraps: bool,
) -> _ActualModuleMeta:
	del init, post_init

	if wraps:
		field_names = _wrapper_field_names
	else:
		field_names = {field.name for field in dataclasses.fields(cls)}

	class _InitableModule(_Initable, cls):
		pass

	def __setattr__(self, name, value):
		if name in field_names:
			if isinstance(value, BoundMethod) and value.__self__ is self:
				raise ValueError()
			else:
				object.__setattr__(self, name, value)
		else:
			raise AttributeError(f"Cannot set attribute {name}")

	_InitableModule.__setattr__ = __setattr__
	_InitableModule.__init__ = cls.__init__
	_InitableModule.__name__ = cls.__name__
	_InitableModule.__qualname__ = cls.__qualname__
	_InitableModule.__module__ = cls.__module__

	return _InitableModule


def _make_initable_wrapper(cls: _ActualModuleMeta) -> _ActualModuleMeta:
	post_init = getattr(cls, "__post_init__", None)
	return _make_initable(cls, cls.__init__, post_init, wraps=False)


def _process_annotation(annotation):
	if isinstance(annotation, str):
		if annotation.startswith("AbstractVar[") or annotation.startswith(
			"AbstractClassVar["
		):
			raise NotImplementedError("Stringified abstract annotations are not supported")
		else:
			is_abstract = False
			is_class = annotation.startswith("ClassVar[")
			return is_abstract, is_class
	else:
		if annotation in (AbstractVar, AbstractClassVar):
			raise TypeError("Cannot use unsubscripted `AbstractVar` or `AbstractClassVar`.")
		elif tp.get_origin(annotation) is AbstractVar:
			if len(tp.get_args(annotation)) != 1:
				raise TypeError("`AbstractVar` can only have a single argument.")
			is_abstract = True
			is_class = False
		elif tp.get_origin(annotation) is AbstractClassVar:
			if len(tp.get_args(annotation)) != 1:
				raise TypeError("`AbstractClassVar` can only have a single argument.")
			is_abstract = True
			is_class = True
		elif tp.get_origin(annotation) is tp.ClassVar:
			is_abstract = False
			is_class = True
		else:
			is_abstract = False
			is_class = False
		return is_abstract, is_class


class ABCMeta(abc.ABCMeta):
	def register(cls, subclass):
		raise ValueError

	def __new__(mcs, name, bases, namespace, /, **kwargs):
		cls = super().__new__(mcs, name, bases, namespace, **kwargs)

		abstract_vars = set()
		abstract_class_vars = set()
		for kls in reversed(cls.__mro__):
			ann = kls.__dict__.get("__annotations__", {})
			for name, annotation in ann.items():
				is_abstract, is_class = _process_annotation(annotation)
				if is_abstract:
					if is_class:
						if name in kls.__dict__:
							raise TypeError(f"Abstract class attribute {name} cannot have value")
						abstract_vars.discard(name)
						abstract_class_vars.add(name)
					else:
						if name in kls.__dict__:
							raise TypeError(f"Abstract attribute {name} cannot have value")
						if name not in abstract_class_vars:
							abstract_vars.add(name)
				else:
					abstract_vars.discard(name)
					if is_class:
						abstract_class_vars.discard(name)
			for name in kls.__dict__.keys():
				abstract_vars.discard(name)
				abstract_class_vars.discard(name)
		cls.__abstractvars__ = frozenset(abstract_vars)
		cls.__abstractclassvars__ = frozenset(abstract_class_vars)
		return cls

	def __call__(cls, *args, **kwargs):
		__tracebackhide__ = True
		if len(cls.__abstractclassvars__) > 0:
			abstract_class_vars = set(cls.__abstractclassvars__)
			raise TypeError(
				f"Can't instantiate abstract class {cls.__name__} with abstract class "
				f"attributes {abstract_class_vars}"
			)
		self = super().__call__(*args, **kwargs)
		if len(cls.__abstractvars__) > 0:
			abstract_class_vars = set(cls.__abstractvars__)
			raise TypeError(
				f"Can't instantiate abstract class {cls.__name__} with abstract "
				f"attributes {abstract_class_vars}"
			)
		return self


@dataclass()
class _FlattenedData:
	dynamic_field_names: tuple
	static_field_names: tuple
	static_field_values: tuple
	wrapper_field_names: tuple
	wrapper_field_values: tuple

	def __repr__(self):
		x = (
			self.dynamic_field_names,
			self.static_field_names,
			self.static_field_values,
		)
		return repr(x)[1:-1]


def _flatten_module(module: "PyTree", with_keys: bool):
	dynamic_field_names = []
	dynamic_field_values = []
	static_field_names = []
	static_field_values = []
	wrapper_field_names = []
	wrapper_field_values = []

	for field_ in dataclasses.fields(module):
		name = field_.name
		try:
			value = module.__dict__[name]
		except KeyError:
			continue
		if field_.metadata.get("static", False):
			static_field_names.append(name)
			static_field_values.append(value)
		else:
			dynamic_field_names.append(name)
			if with_keys:
				dynamic_field_values.append((tu.GetAttrKey(name), value))
			else:
				dynamic_field_values.append(value)
	sentinel = object()
	for name in _wrapper_field_names:
		value = getattr(module, name, sentinel)
		if value is not sentinel:
			wrapper_field_names.append(name)
			wrapper_field_values.append(value)
	aux = _FlattenedData(
		tuple(dynamic_field_names),
		tuple(static_field_names),
		tuple(static_field_values),
		tuple(wrapper_field_names),
		tuple(wrapper_field_values),
	)
	return tuple(dynamic_field_values), aux


def _unflatten_module(cls: type["PyTree"], aux: _FlattenedData, dynamic_field_values):
	module = object.__new__(cls)
	for name, value in zip(aux.dynamic_field_names, dynamic_field_values):  # noqa
		object.__setattr__(module, name, value)
	for name, value in zip(aux.static_field_names, aux.static_field_values):  # noqa
		object.__setattr__(module, name, value)
	for name, value in zip(aux.wrapper_field_names, aux.wrapper_field_values):  # noqa
		object.__setattr__(module, name, value)
	return module


@dataclass(frozen=True)
class StrictConfig:
	force_abstract: bool = False
	allow_abstract_name: bool = False
	allow_method_override: bool = False


class _wrap_method:
	def __init__(self, method):
		self.method = method
		if getattr(self.method, "__isabstractmethod__", False):
			self.__isabstractmethod__ = self.method.__isabstractmethod__

	def __get__(self, instance, owner):
		if instance is None:
			return self.method
		else:
			_method = _module_update_wrapper(
				BoundMethod(self.method, instance),
				None,
				inplace=True,
			)
			return _method


class _ActualModuleMeta(ABCMeta):
	def __new__(
		mcs,
		name: str,
		bases: tuple,
		class_dict: dict,
		/,
		strict: tp.Union[bool, "StrictConfig"] = False,
		**kwargs,
	):
		# --- Configure strict mode ---
		if isinstance(strict, bool):
			strict_config = StrictConfig()
		elif isinstance(strict, StrictConfig):
			strict_config = strict
			strict = True
		else:
			raise TypeError("`strict` must be a bool or StrictConfig instance")

		# --- Create the class ---
		cls = super().__new__(mcs, name, bases, class_dict, **kwargs)

		# --- Wrap all non-magic methods ---
		for attr_name, attr_value in cls.__dict__.items():
			if _not_magic(attr_name) and inspect.isfunction(attr_value):
				setattr(cls, attr_name, _wrap_method(attr_value))

		# --- Determine if a custom __init__ is provided and if dataclass init is active ---
		added_custom_init = "__init__" in cls.__dict__
		has_dataclass_init = False
		if added_custom_init:
			has_dataclass_init = False
		else:
			# Check bases for dataclass __init__ presence.
			for base in cls.__mro__[1:-1]:
				try:
					has_dataclass_init = _has_dataclass_init[base]
				except KeyError:
					if base.__init__ is not object.__init__:
						has_dataclass_init = False
						break
				else:
					break
			else:
				# Special case: The PyTree base should always have a dataclass __init__
				assert name == "PyTree"
				has_dataclass_init = True

		# --- Wrap __post_init__ to convert fields if needed ---
		if has_dataclass_init and "__post_init__" in cls.__dict__:
			original_post_init = cls.__post_init__

			@ft.wraps(original_post_init)
			def __post_init__(self, *args, **kwargs):
				if self.__class__ is _make_initable_wrapper(cls):
					_convert_fields(self, init=True)
				original_post_init(self, *args, **kwargs)
				if self.__class__ is _make_initable_wrapper(cls):
					_convert_fields(self, init=False)

			cls.__post_init__ = __post_init__
		else:
			original_post_init = None

		# --- Preserve __init__ docstring if using dataclass init ---
		if has_dataclass_init:
			init_doc = cls.__init__.__doc__

		# --- Convert class to a dataclass ---
		cls = dataclasses.dataclass(
			eq=False,
			repr=False,
			frozen=True,
			init=has_dataclass_init,
		)(cls)

		# --- Update __init__ annotations based on field metadata ---
		for field in dataclasses.fields(cls):
			if field.name not in cls.__init__.__annotations__:
				continue
			try:
				converter = field.metadata["converter"]
			except KeyError:
				continue

			try:
				signature = inspect.signature(converter)
			except ValueError:
				converter_annotation = tp.Any
			else:
				params = list(signature.parameters.values())
				if not params:
					converter_annotation = tp.Any
				else:
					converter_annotation = params[0].annotation
					if converter_annotation is inspect.Signature.empty:
						converter_annotation = tp.Any
			cls.__init__.__annotations__[field.name] = converter_annotation

		_has_dataclass_init[cls] = has_dataclass_init

		# --- Wrap __init__ if __post_init__ was not redefined ---
		if original_post_init is None:
			original_init = cls.__init__

			@ft.wraps(original_init)
			def __init__(self, *args, **kwargs):
				__tracebackhide__ = True  # Hide traceback details.
				original_init(self, *args, **kwargs)
				if self.__class__ is _make_initable_wrapper(cls):
					_convert_fields(self, init=True)
					_convert_fields(self, init=False)

			cls.__init__ = __init__
		else:
			# Restore original __init__ docstring and module attributes.
			cls.__init__.__doc__ = init_doc
			cls.__init__.__module__ = cls.__module__

		# --- Record strict mode attributes ---
		_is_force_abstract[cls] = strict_config.force_abstract
		_is_strict[cls] = strict

		# --- Validate base classes and method overrides in strict mode ---
		if strict:
			for base in bases:
				if base is PyTree:
					continue
				if _is_special_form(base):
					continue
				if not issubclass(base, PyTree):
					raise TypeError(f"Base class {base} must be a subclass of PyTree")
				if not _is_strict[base]:
					raise TypeError("All base classes must be strict")
				if not _is_abstract(base):
					raise TypeError("All base classes must be abstract")
				base_field_count = len(dataclasses.fields(base))
				if (base_field_count > 0) or (not _has_dataclass_init[base]):
					if len(dataclasses.fields(cls)) != base_field_count:
						raise TypeError("Field count mismatch between base and derived class")
					if added_custom_init:
						raise TypeError("Custom __init__ not allowed in strict mode")

			# --- Enforce naming conventions for abstract classes ---
			if not strict_config.allow_abstract_name:
				has_abstract_prefix = cls.__name__.startswith(
					"Abstract"
				) or cls.__name__.startswith("_Abstract")
				if _is_abstract(cls):
					if not has_abstract_prefix:
						if _is_force_abstract[cls]:
							raise TypeError("Abstract class must have an abstract name")
						details = []
						if cls.__abstractmethods__:
							details.append(f"abstract methods: {list(cls.__abstractmethods__)}")
						if getattr(cls, "__abstractvars__", []):
							details.append(f"abstract variables: {list(cls.__abstractvars__)}")
						if getattr(cls, "__abstractclassvars__", []):
							details.append(
								f"abstract class variables: {list(cls.__abstractclassvars__)}"
							)
						detail_str = ", ".join(details)
						raise TypeError(
							f"Abstract class missing proper naming. Details: {detail_str}"
						)
				else:
					if has_abstract_prefix:
						raise TypeError("Non-abstract class cannot have an abstract name")

			# --- Validate method overrides ---
			for attr_name, attr_value in cls.__dict__.items():
				if isinstance(attr_value, _wrap_method):
					method = attr_value.method
					if not getattr(method, "__isabstractmethod__", False):
						for base in bases:
							base_method = getattr(base, attr_name, _dummy_abstract)
							if not inspect.isfunction(base_method):
								raise TypeError("Method override type mismatch")
							if not strict_config.allow_method_override and not getattr(
								base_method, "__isabstractmethod__", False
							):
								raise TypeError("Method override is not allowed in strict mode")

		# --- Register pytree flatten/unflatten functions ---
		tu.register_pytree_with_keys(
			cls,
			flatten_with_keys=ft.partial(_flatten_module, with_keys=True),
			flatten_func=ft.partial(_flatten_module, with_keys=False),
			unflatten_func=ft.partial(_unflatten_module, cls),
		)
		return cls

	@property
	def __signature__(cls):
		# Return the signature of __init__ excluding the 'self' parameter.
		init_sig = inspect.signature(cls.__init__)
		params = list(init_sig.parameters.values())[1:]
		return init_sig.replace(parameters=params)

	def __call__(cls, *args, **kwargs):
		__tracebackhide__ = True  # Hide internal traceback details.
		if _is_force_abstract[cls]:
			raise TypeError("Cannot instantiate a forced abstract class.")

		# Use the initable wrapper for instantiation.
		initable_cls = _make_initable_wrapper(cls)
		instance = super(_ActualModuleMeta, initable_cls).__call__(*args, **kwargs)

		# Ensure all dataclass fields are initialized.
		missing_fields = {
			field.name for field in dataclasses.fields(cls) if field.name not in dir(instance)
		}
		if missing_fields:
			raise ValueError(
				f"The following fields were not initialised during __init__: {missing_fields}"
			)

		# Warn if static fields hold JAX arrays.
		for field in dataclasses.fields(instance):
			if field.metadata.get("static", False):
				field_value = getattr(instance, field.name)
				if any(
					tu.tree_map(lambda x: isinstance(x, Array), tu.tree_flatten(field_value)[0])
				):
					warnings.warn(
						"A JAX array is being set as static! This can result in unexpected "
						"behavior and is usually a mistake.",
						stacklevel=2,
					)

		# Set the correct __class__ on the instance.
		object.__setattr__(instance, "__class__", cls)

		# Run any __check_init__ methods defined in the class hierarchy.
		for base in cls.__mro__:
			check_init = base.__dict__.get("__check_init__")
			if check_init is not None:
				check_init(instance)

		return instance

	def __getattribute__(cls, item):
		value = super().__getattribute__(item)
		if (
			item == "__wrapped__"
			and isinstance(value, property)
			and cls not in _has_dataclass_init
		):
			raise AttributeError
		return value

	def __setattr__(cls, item, value):
		# Wrap function assignments to enforce proper behavior.
		if _not_magic(item) and inspect.isfunction(value):
			value = _wrap_method(value)
		super().__setattr__(item, value)


if tp.TYPE_CHECKING:

	@dataclass_transform(field_specifiers=(dataclasses.field, field, static_field))
	class _ModuleMeta(abc.ABCMeta):
		__abstractvars__: frozenset[str]
		__abstractclassvars__: frozenset[str]
else:
	_ModuleMeta = _ActualModuleMeta


class PyTree(metaclass=_ModuleMeta):
	def __hash__(self):
		return hash(tuple(tu.tree_leaves(self)))

	def __eq__(self, other) -> tp.Union[bool, np.bool_, Array]:
		return tree_equal(self, other)

	def __repr__(self):
		return tree_pformat(self)


class Partial(PyTree):
	func: tp.Callable
	args: tuple[tp.Any, ...]
	keywords: dict[str, tp.Any]

	def __init__(self, func, /, *args, **kwargs):
		self.func = func
		self.args = args
		self.keywords = kwargs

	def __call__(self, *args, **kwargs):
		return self.func(*self.args, *args, **kwargs, **self.keywords)


class BoundMethod(PyTree):
	__func__: types.FunctionType = field(static=True)
	__self__: PyTree

	def __call__(self, *args, **kwargs):
		__tracebackhide__ = True
		return self.__func__(self.__self__, *args, **kwargs)

	@property
	def __wrapped__(self):
		return self.__func__.__get__(self.__self__, type(self.__self__))

	@property
	def __signature__(self):
		return inspect.signature(self.__wrapped__)


def module_update_wrapper(
	wrapper: PyTree,
	wrapped: tp.Optional[tp.Callable[_P, _T]] = None,
) -> tp.Callable[_P, _T]:
	return _module_update_wrapper(
		wrapper,
		wrapped,
		inplace=False,
	)


def _module_update_wrapper(
	wrapper: PyTree,
	wrapped: tp.Optional[tp.Callable[_P, _T]],
	inplace: bool,
) -> tp.Callable[_P, _T]:
	cls = wrapper.__class__
	if not isinstance(getattr(cls, "__wrapped__", None), property):
		raise ValueError("Wrapper module must supply `__wrapped__` as a property.")

	if wrapped is None:
		wrapped = wrapper.__wrapped__

	if not inplace:
		leaves, treedef = tu.tree_flatten(wrapper)
		wrapper = tu.tree_unflatten(treedef, leaves)

	initable_cls = _make_initable(cls, None, None, wraps=True)
	object.__setattr__(wrapper, "__class__", initable_cls)
	try:
		for attr in _wrapper_field_names:
			try:
				value = getattr(wrapped, attr)
			except AttributeError:
				pass
			else:
				setattr(wrapper, attr, value)
	finally:
		object.__setattr__(wrapper, "__class__", cls)
	return tp.cast(tp.Callable[_P, _T], wrapper)
