from dataclasses import dataclass  # noqa

import jax
import jax.random
import jax.tree_util
import optax
import os
import sys
from tqdm import tqdm

jax.config.update("jax_platform_name", "cpu")


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__),
        ),
        "../src",
    )
)

import fjformer.core as core  # noqa
from fjformer import GenerateRNG  # noqa
from jax import numpy as jnp  # noqa
from fjformer.lora import LoraRapture, RaptureConfig  # noqa
from flax import linen as nn  # noqa
from fjformer.optimizers.adamw import get_adamw_with_warmup_cosine_scheduler  # noqa

rng = GenerateRNG()
STEPS = 50_000


class Model(nn.Module):
    """A simple linear model for demonstration."""

    fc_out: int = 6
    fc1_out: int = 4
    out_num: int = 1

    def setup(self) -> None:
        """Initializes the model layers."""
        self.fc = nn.Dense(self.fc_out, use_bias=False, dtype=jnp.float32)
        self.fc1 = nn.Dense(self.fc1_out, use_bias=False, dtype=jnp.float32)
        self.out = nn.Dense(self.out_num, use_bias=False, dtype=jnp.float32)

    def __call__(self, x):
        """Performs a forward pass through the model."""
        x = self.fc(x)
        x = self.fc1(x)
        return self.out(x)


def main():
    """
    Demonstrates training a model with LoraRapture using the AdamW optimizer with a warmup cosine scheduler.
    - Initializes a model, LoraRapture, and optimizer.
    - Defines a JIT-compiled training function.
    - Trains the model for a specified number of steps, printing progress and loss.
    """
    model = Model()
    init_x = jax.random.normal(rng.rng, (1, 64))
    rapture = LoraRapture(
        config=RaptureConfig(
            lora_dim=64,
            lora_fine_tune_parameters=["fc", "out"],
        )
    )
    tx, scheduler = get_adamw_with_warmup_cosine_scheduler(STEPS, 3e-4)
    out = rapture.apply_lora(
        module=model,
        parameters=model.init(rng.rng, init_x),
        tx=tx,
        decision_fn=None,
        tune_vectors=False,
        rng=rng.rng,
        func_target="apply",
    )
    lora_model = out.lora_module
    lora_opt_state = out.lora_opt_state
    lora_parameters = out.lora_parameters
    lora_tx = out.lora_tx

    @jax.jit
    def train_fn(
        lora_params,
        opt_state,
        inputs,
    ):
        """
        Defines the training step for the model.

        Args:
            lora_params: Lora parameters.
            opt_state: Optimizer state.
            inputs: Input data.

        Returns:
            Loss, new optimizer state, and updated Lora parameters.
        """

        def loss_fn(run_parameters):
            out = lora_model(
                run_parameters,
                inputs,
            )
            return (2 ** (out - (inputs + 1))).sum().mean()

        loss, grad = jax.value_and_grad(loss_fn)(lora_params)

        updates, new_opt_state = lora_tx.update(grad, opt_state, params=lora_params)
        updated_params = optax.apply_updates(lora_params, updates)
        return loss, new_opt_state, updated_params

    bar = tqdm(range(STEPS))
    for index in bar:
        loss, lora_opt_state, lora_parameters = train_fn(
            lora_parameters,
            lora_opt_state,
            jax.random.normal(rng.rng, (1, 64)),
        )
        bar.set_postfix(
            step=index,
            learning_rate=f"{scheduler(index):.5e}",
            loss=f"{loss:.3e}",
        )
        bar.update(1)
    bar.close()


if __name__ == "__main__":
    main()
