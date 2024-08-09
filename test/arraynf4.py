import os
import sys
from typing import Any, Callable, Literal, Optional

import flax.traverse_util
import jax
import jax.random
import jax.tree_util
from jax import Array

jax.config.update("jax_platform_name", "cpu")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
import fjformer.core as core  # noqa
from fjformer import GenerateRNG  # noqa
from jax import numpy as jnp  # noqa
from fjformer.custom_array.arraynf4 import ArrayNF4  # noqa
from flax import linen as nn  # noqa
from fjformer.core.implicit_array import implicit_compact  # noqa


rng = GenerateRNG()


class Model(nn.Module):
    """A simple linear model for demonstration."""

    def setup(self) -> None:
        """Initializes the model layers."""
        self.embed_time = nn.Embed(512, 512)
        self.fc = nn.Dense(512, use_bias=True, dtype=jnp.float32)
        self.fc1 = nn.Dense(64, use_bias=True, dtype=jnp.float32)
        self.out = nn.Dense(1, use_bias=True, dtype=jnp.float32)

    def __call__(self, x):
        """Performs a forward pass through the model."""
        x_time = jnp.arange(x.shape[1]).reshape(x.shape[0], -1)
        xt = self.embed_time(x_time)
        x = self.fc(x) + xt
        x = self.fc1(x)
        return self.out(x)


def quantize_params(
    params: dict,
    block_size: Literal[32, 64, 128, 256, 512, 1024, 2048, 4096] = 64,
    contraction_axis: int = -1,
    factors: Optional[Array] = None,
) -> dict:
    """Quantizes model parameters using ArrayNF4.

    Args:
        params: A dictionary of model parameters.

    Returns:
        A dictionary of quantized model parameters.
    """

    def q(path: str, array: Any) -> ArrayNF4:
        """Quantizes a single parameter array."""
        path = ".".join(p for p in path[0].key)
        if path.endswith(".embedding"):
            return array
        if array.ndim > 2:
            return array
        dim = array.shape[contraction_axis]
        if not (dim % 2 == 0):
            return array
        print(f"{path} quantized.")
        return ArrayNF4.quantize(
            array=array,
            block_size=block_size,
            contraction_axis=contraction_axis,
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
    init_x = jax.random.normal(rng.rng, (1, 1, 64))
    x = jax.random.normal(rng.rng, (1, 1, 64))
    params = model.init(rng.rng, init_x)
    model_apply: Callable = jax.jit(implicit_compact(model.apply))
    q_params = quantize_params(params)
    q_out = float(model_apply(q_params, x).reshape(-1)[0])
    out = float(model_apply(params, x).reshape(-1)[0])
    print(f"Original Model  Output: {out:.3e}")
    print(f"Quantized Model Output: {q_out:.3e}")
    print(f"Overall error: {(out-q_out):.5e}")


if __name__ == "__main__":
    main()
