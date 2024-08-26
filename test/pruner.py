import os
import sys

import jax
import jax.random
import jax.tree_util

jax.config.update("jax_platform_name", "cpu")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
import functools

import jax
import jax.numpy as jnp
import optax
from fjformer import GenerateRNG, jaxpruner
from fjformer.jaxpruner import sparsity_distributions, sparsity_schedules
from flax import linen as nn
import matplotlib.pyplot as plt

rng = GenerateRNG()


class Model(nn.Module):
	"""A simple linear model for demonstration."""

	def setup(self) -> None:
		"""Initializes the model layers."""
		self.fc = nn.Dense(512, use_bias=False, dtype=jnp.float32)
		self.fc1 = nn.Dense(1024, use_bias=False, dtype=jnp.float32)
		self.fc2 = nn.Dense(256, use_bias=False, dtype=jnp.float32)
		self.out = nn.Dense(1, use_bias=False, dtype=jnp.float32)

	def __call__(self, x):
		"""Performs a forward pass through the model."""
		x = self.fc(x) + x
		x = self.fc2(self.fc1(x))
		return self.out(x)


def main():
	model = Model()
	params = model.init(
		jax.random.PRNGKey(0),
		x=jax.random.normal(
			rng.rng,
			(1, 512),
		),
	)
	pruner = jaxpruner.MagnitudePruning(
		sparsity_distribution_fn=functools.partial(
			sparsity_distributions.uniform,
			sparsity=0.8,
		),
		scheduler=sparsity_schedules.OneShotSchedule(target_step=0),
	)
	tx = optax.adamw(3e-9)
	# tx = pruner.wrap_optax(tx)
	opt_state = tx.init(params=params)

	@jax.jit
	def forward(params, opt_state, x, y):
		def loss_fn(p):
			return jnp.sum((y - model.apply(p, x)) ** 2)

		loss, grad = jax.value_and_grad(loss_fn)(params)
		updates, opt_state = tx.update(grad, opt_state, params)
		params = optax.apply_updates(params=params, updates=updates)
		return params, opt_state, loss

	num_steps = 2048
	loss_history = []
	for i in range(1, num_steps + 1):
		x = jax.random.uniform(rng.rng, (64, 512))
		params, opt_state, loss = forward(
			params=params,
			opt_state=opt_state,
			x=x,
			y=x[:, -1],
		)
		if hasattr(tx, "__pruner_tx__"):
			params = pruner.post_gradient_update(params=params, sparse_state=opt_state)
		if i % 5 == 0:
			print(f"\rEpoch : {i} | loss {loss}", end="")
		loss_history.append(loss)

	plt.plot(loss_history)
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.title("Training Loss")
	plt.savefig(f"tmp_{hasattr(tx, '__pruner_tx__')}.png")


if __name__ == "__main__":
	main()
