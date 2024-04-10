import jax.random
from jax import numpy as jnp

import src.fjformer.linen as nn
from src.fjformer import GenerateRNG


class DummyNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.LayerNorm(

        )(
            nn.Linear(
                2
            )(
                nn.Embed(
                    512, 1024
                )(
                    x
                )
            )
        )


def main():
    rng_gen = GenerateRNG(42)
    net = DummyNet()

    params = net.init(
        rng_gen.rng,
        jax.random.randint(rng_gen.rng, (1, 68), minval=0, maxval=512)
    )

    quantized_params = nn.quantize_params(params)

    inputs = jax.random.randint(rng_gen.rng, (1, 1), minval=0, maxval=512)

    org_pred = net.apply(params, inputs)
    qun_pred = net.apply(quantized_params, inputs)
    print(jnp.allclose(org_pred, qun_pred, rtol=1e-2, atol=1e-8))
    print(org_pred)
    print(qun_pred)


if __name__ == '__main__':
    main()
