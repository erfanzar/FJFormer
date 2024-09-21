import os
import sys
from typing import Any, Callable, Literal, Optional

import jax
import jax.random
import jax.tree_util

jax.config.update("jax_platform_name", "cpu")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
from fjformer import GenerateRNG
from fjformer.core.implicit_array import implicit_compact
from fjformer.dtypes.arraynf4 import ArrayNF4
from flax import linen as nn
from jax import numpy as jnp

rng = GenerateRNG()


class Model(nn.Module):
	"""A simple linear model for demonstration."""

	inx: int

	def setup(self) -> None:
		"""Initializes the model layers."""
		self.embed_time = nn.Embed(512, self.inx)
		self.fc = nn.Dense(
			self.inx,
			use_bias=False,
			dtype=jnp.float32,
			kernel_init=jax.nn.initializers.normal(0.02),
		)
		self.fc1 = nn.Dense(
			self.inx // 2,
			use_bias=False,
			dtype=jnp.float32,
			kernel_init=jax.nn.initializers.normal(0.02),
		)
		self.out = nn.Dense(
			self.inx // 5,
			use_bias=False,
			dtype=jnp.float32,
			kernel_init=jax.nn.initializers.normal(0.02),
		)

	def __call__(self, x):
		"""Performs a forward pass through the model."""
		x_time = jnp.arange(x.shape[1]).reshape(x.shape[0], -1)
		xt = self.embed_time(x_time)
		x = self.fc(x) + xt
		x = self.fc1(x)
		return self.out(x)


def quantize_params(params: dict) -> dict:
	"""Quantizes model parameters using ArrayNF4.

	Returns:
	    A dictionary of quantized model parameters.
	"""

	def q(path, array: Any) -> ArrayNF4:
		"""Quantizes a single parameter array."""
		if array.ndim > 2 or array.size < 128 or path[-1].key == "embedding":
			return array
		return ArrayNF4.quantize(array=array)

	return jax.tree_util.tree_map_with_path(q, params)


def main():
	"""
	Demonstrates the quantization process using a simple model.
	- Initializes a model and random input data.
	- Quantizes the model parameters.
	- Performs inference using both the original and quantized models.
	- Prints the output of both models for comparison.
	"""
	model = Model(inx=512)
	init_x = jax.random.normal(rng.rng, (1, 1, 64))
	x = jax.random.normal(rng.rng, (1, 1, 64))
	params = model.init(rng.rng, init_x)
	model_apply: Callable = jax.jit(implicit_compact(model.apply))
	q_params = quantize_params(params)
	q_out = model_apply(q_params, x)
	q_out = model_apply(q_params, x)
	q_out = model_apply(q_params, x)
	with jax.profiler.trace("/tmp/somes"):
		q_out = model_apply(q_params, x)
		out = model_apply(params, x)
	error = jnp.abs(q_out - out).max()
	print(f"ERROR : {error}")
	jax.profiler.save_device_memory_profile("mem.prof")


if __name__ == "__main__":
	main()
