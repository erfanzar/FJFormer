import os
import sys
from typing import Any, Callable

import flax.traverse_util
import jax
import jax.random
import jax.tree_util

jax.config.update("jax_platform_name", "cpu")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
import fjformer.core as core  # noqa
from fjformer import GenerateRNG  # noqa
from jax import numpy as jnp  # noqa
from fjformer.custom_array.array8lt import Array8Lt  # noqa
from flax import linen as nn  # noqa
from fjformer.core.implicit_array import implicit_compact  # noqa


rng = GenerateRNG()


class Model(nn.Module):
    """A simple linear model for demonstration."""

    def setup(self) -> None:
        """Initializes the model layers."""
        self.fc = nn.Dense(512, use_bias=True, dtype=jnp.float32)
        self.fc1 = nn.Dense(64, use_bias=True, dtype=jnp.float32)
        self.out = nn.Dense(1, use_bias=True, dtype=jnp.float32)

    def __call__(self, x):
        """Performs a forward pass through the model."""
        x = self.fc(x)
        x = self.fc1(x)
        return self.out(x)


def quantize_params(params: dict) -> dict:
    """Quantizes model parameters using Array8Lt.

    Args:
        params: A dictionary of model parameters.

    Returns:
        A dictionary of quantized model parameters.
    """

    def q(path: str, array: Any) -> Array8Lt:
        """Quantizes a single parameter array."""
        path = ".".join(p for p in path[0].key)
        return Array8Lt.quantize(
            array,
        )

    return flax.traverse_util.unflatten_dict(
        jax.tree_util.tree_map_with_path(
            q,
            flax.traverse_util.flatten_dict(params),
        )
    )


def main():
    """
    Demonstrates the quantization process using a simple model.
    - Initializes a model and random input data.
    - Quantizes the model parameters.
    - Performs inference using both the original and quantized models.
    - Prints the output of both models for comparison.
    """
    model = Model()
    init_x = jax.random.normal(rng.rng, (1, 64))
    x = jax.random.normal(rng.rng, (1, 64))
    params = model.init(rng.rng, init_x)
    q_params = quantize_params(params)
    q_model: Callable = jax.jit(implicit_compact(model.apply))
    q_out = float(q_model(q_params, x).reshape(-1)[0])
    out = float(model.apply(params, x).reshape(-1)[0])
    print(f"Original Model  Output: {out:.3e}")
    print(f"Quantized Model Output: {q_out:.3e}")
    print(f"Overall error: {(out-q_out):.5e}")


if __name__ == "__main__":
    main()
