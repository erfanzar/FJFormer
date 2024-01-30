import flax.core
import jax.tree_util
import termcolor

from .lora_transform import (
    LORA_FULL,
    LORA_FREEZE,
    apply_lora_with_implicit_args,
    wrap_optimizer,
    make_lora_spec
)
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class XRabTurConfig:
    lora_dim: int
    fully_fine_tune_parameters: Optional[list[str]] = None
    lora_fine_tune_parameters: Optional[list[str]] = None
    tune_vectors: bool = True
    verbose: bool = True


class XRabTur:
    def __init__(
            self,
            config: XRabTurConfig
    ):
        self.config = config

    def base_decision_function(
            self,
            path: list[jax.tree_util.DictKey],
            params: dict | jax.tree_util.PyTreeDef | None = None
    ):
        if self.config.fully_fine_tune_parameters is not None:
            for param_name in self.config.fully_fine_tune_parameters:
                if jax.tree_util.DictKey(key=param_name) in path:
                    if self.config.verbose:
                        print(
                            termcolor.colored(
                                f"Parameter"
                                f" {'/'.join(str(n.key) for n in path)}"
                                f"Selected for Fully Fine-Tune.",
                                color="cyan",
                                force_color=True
                            )
                        )
                    return LORA_FULL

        if self.config.lora_fine_tune_parameters is not None:
            for param_name in self.config.lora_fine_tune_parameters:
                if jax.tree_util.DictKey(key=param_name) in path:
                    if self.config.verbose:
                        print(
                            termcolor.colored(
                                f"Parameter"
                                f" {'/'.join(str(n.key) for n in path)}"
                                f"Selected for LoRA Fine-Tune with {self.config.lora_dim} dimensions.",
                                color="cyan",
                                force_color=True
                            )
                        )
                    return self.config.lora_dim

        return LORA_FREEZE

    def _make_lora_specs(
            self,
            parameters: dict | flax.core.FrozenDict,
            decision_fn: Optional[Callable] = None,
            tune_vectors: bool = False,
    ):
        return make_lora_spec(
            params=parameters,
            decision_fn=decision_fn if decision_fn is not None else self.base_decision_function,
            tune_vectors=tune_vectors
        )
