import jax.random
from jax import numpy as jnp
import src.fjformer.linen as nn
from src.fjformer import GenerateRNG


def main():
    rng_gen = GenerateRNG(42)
    neuron = nn.Linear(
        4,
        use_bias=True
    )

    params = neuron.init(
        rng_gen.rng,
        jax.random.normal(rng_gen.rng, (1, 68, 4))
    )

    quantized_params = nn.quantize_params(params)

    inputs = jax.random.normal(rng_gen.rng, (1, 1, 4))

    org_pred = neuron.apply(params, inputs)
    qun_pred = neuron.apply(quantized_params, inputs)
    print(jnp.allclose(org_pred, qun_pred, rtol=1e-2, atol=1e-8))


if __name__ == '__main__':
    main()
