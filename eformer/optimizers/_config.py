# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import json
import typing as tp
import warnings
from dataclasses import asdict, dataclass, fields

import jax.numpy as jnp

T = tp.TypeVar("T", bound="SerializationMixin")


class SerializationMixin:
	"""
	Mixin class providing serialization capabilities for configuration classes.

	This class provides methods to convert instances to and from dictionaries and JSON strings,
	making it easy to serialize and deserialize configuration objects.

	Methods:
	    to_dict: Convert the instance to a dictionary, filtering out private fields.
	    from_dict: Create an instance from a dictionary with error checking.
	    to_json: Serialize the instance to a JSON string.
	    from_json: Create an instance from a JSON string.
	"""

	def to_dict(self) -> tp.Dict[str, tp.Any]:
		"""
		Convert the instance to a dictionary, filtering out private fields.

		Returns:
		    dict: A dictionary representation of the instance, excluding private fields.
		"""
		return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

	@classmethod
	def from_dict(cls: tp.Type[T], data: tp.Dict[str, tp.Any]) -> T:
		"""
		Create an instance from a dictionary with error checking.

		Args:
		    data (dict): A dictionary containing the data to populate the instance.

		Returns:
		    T: An instance of the class populated with the provided data.

		Raises:
		    Warning: If unexpected keys are present in the input dictionary.
		"""
		valid_fields = {f.name for f in fields(cls)}
		extra_keys = set(data.keys()) - valid_fields
		if extra_keys:
			warnings.warn(
				f"Ignoring unexpected keys {extra_keys} for {cls.__name__}",
				stacklevel=2,
			)

		filtered = {k: v for k, v in data.items() if k in valid_fields}
		return cls(**filtered)

	def to_json(self) -> str:
		"""
		Serialize the instance to a JSON string.

		Returns:
		    str: A JSON string representation of the instance.
		"""
		return json.dumps(self.to_dict(), indent=2)

	@classmethod
	def from_json(cls: tp.Type[T], json_str: str) -> T:
		"""
		Create an instance from a JSON string.

		Args:
		    json_str (str): A JSON string containing the data to populate the instance.

		Returns:
		    T: An instance of the class populated with the data from the JSON string.
		"""
		return cls.from_dict(json.loads(json_str))


@dataclass
class SchedulerConfig(SerializationMixin):
	"""
	Configuration class for learning rate schedulers.

	Attributes:
	    scheduler_type (Optional[Literal["linear", "cosine"]]): Type of scheduler to use.
	    learning_rate (float): Initial learning rate. Defaults to 5e-5.
	    learning_rate_end (Optional[float]): Final learning rate for linear scheduler.
	    warmup_steps (Optional[int]): Number of warmup steps.
	    steps (Optional[int]): Total number of steps. Required for non-constant schedulers.
	    exponent (float): Exponent for polynomial decay. Defaults to 1.0.

	Methods:
	    __post_init__: Validates the configuration after initialization.
	    _validate: Performs validation checks on the configuration.
	"""

	scheduler_type: tp.Optional[tp.Literal["linear", "cosine"]] = None
	learning_rate: float = 5e-5
	learning_rate_end: tp.Optional[float] = None
	warmup_steps: tp.Optional[int] = None
	steps: tp.Optional[int] = None  # Required for non-constant schedulers
	exponent: float = 1.0

	def __post_init__(self):
		"""
		Validates the configuration after initialization.
		"""
		self._validate()

	def _validate(self):
		"""
		Performs validation checks on the configuration.

		Raises:
		    ValueError: If the configuration is invalid.
		"""
		# Validate steps requirements
		if self.scheduler_type is not None and self.steps is None:
			raise ValueError("Steps must be specified for non-constant schedulers")

		# Type-specific validation
		if self.scheduler_type == "linear":
			if self.learning_rate_end is None:
				raise ValueError("Linear scheduler requires learning_rate_end")

		# Warmup validation
		if self.warmup_steps is not None:
			if self.steps is None:
				raise ValueError("Steps required when using warmup")
			if self.warmup_steps >= self.steps:
				raise ValueError("Warmup steps must be less than total steps")


@dataclass
class AdafactorConfig(SerializationMixin):
	"""
	Configuration class for the Adafactor optimizer.

	Attributes:
	    min_dim_size_to_factor (int): Minimum dimension size for factoring. Defaults to 128.
	    decay_rate (float): Decay rate for second-moment estimator. Defaults to 0.8.
	    decay_offset (int): Decay offset. Defaults to 0.
	    multiply_by_parameter_scale (bool): Whether to multiply by parameter scale. Defaults to True.
	    clipping_threshold (Optional[float]): Clipping threshold for updates. Defaults to 1.0.
	    momentum (Optional[float]): Momentum factor. Defaults to None.
	    dtype_momentum (jnp.dtype): Data type for momentum. Defaults to jnp.float32.
	    weight_decay_rate (Optional[float]): Weight decay rate. Defaults to None.
	    eps (float): Small constant for numerical stability. Defaults to 1e-30.
	    factored (bool): Whether to use factored second-moment estimates. Defaults to True.
	"""

	min_dim_size_to_factor: int = 128
	decay_rate: float = 0.8
	decay_offset: int = 0
	multiply_by_parameter_scale: bool = True
	clipping_threshold: tp.Optional[float] = 1.0
	momentum: tp.Optional[float] = None
	dtype_momentum: jnp.dtype = jnp.float32
	weight_decay_rate: tp.Optional[float] = None
	eps: float = 1e-30
	factored: bool = True


@dataclass
class AdamWConfig(SerializationMixin):
	"""
	Configuration class for the AdamW optimizer.

	Attributes:
	    b1 (float): Exponential decay rate for the first moment estimates. Defaults to 0.9.
	    b2 (float): Exponential decay rate for the second moment estimates. Defaults to 0.999.
	    eps (float): Small constant for numerical stability. Defaults to 1e-8.
	    eps_root (float): Small constant for root calculations. Defaults to 0.0.
	    mu_dtype (Optional[jnp.dtype]): Data type for momentum. Defaults to None.
	"""

	b1: float = 0.9
	b2: float = 0.999
	eps: float = 1e-8
	eps_root: float = 0.0
	mu_dtype: tp.Optional[jnp.dtype] = None


@dataclass
class LionConfig(SerializationMixin):
	"""
	Configuration class for the Lion optimizer.

	Attributes:
	    b1 (float): Exponential decay rate for the first moment estimates. Defaults to 0.9.
	    b2 (float): Exponential decay rate for the second moment estimates. Defaults to 0.99.
	    mu_dtype (Optional[jnp.dtype]): Data type for momentum. Defaults to None.
	"""

	b1: float = 0.9
	b2: float = 0.99
	mu_dtype: tp.Optional[jnp.dtype] = None


@dataclass
class RMSPropConfig(SerializationMixin):
	"""
	Configuration class for the RMSProp optimizer.

	Attributes:
	    decay (float): Decay rate for the moving average. Defaults to 0.9.
	    initial_scale (float): Initial scale for the moving average. Defaults to 0.0.
	    momentum (Optional[float]): Momentum factor. Defaults to None.
	    nesterov (bool): Whether to use Nesterov momentum. Defaults to False.
	    eps (float): Small constant for numerical stability. Defaults to 1e-8.
	"""

	decay: float = 0.9
	initial_scale: float = 0.0
	momentum: tp.Optional[float] = None
	nesterov: bool = False
	eps: float = 1e-8
