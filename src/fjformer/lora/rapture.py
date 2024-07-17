from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import flax.core
import flax.traverse_util
import jax
import jax.tree_util
import optax
from jax import numpy as jnp

from fjformer.core import (
    EmptyNode,
    implicit_compact,
    materialize_nested,
    tree_map_with_implicit,
)
from fjformer.core.utilities import freeze_keys, freeze_subtrees
from fjformer.lora.lora_core import LoraWeight
from fjformer.utils import get_logger

LORA_FREEZE = 0
LORA_FULL = -1

logger = get_logger(__name__)


def split_lora_params(
    params: Union[dict, jax.tree_util.PyTreeDef],
    lora_spec: Union[dict, jax.tree_util.PyTreeDef],
) -> Union[dict, jax.tree_util.PyTreeDef]:
    """Replaces specific parameters in a PyTree with `EmptyNode`.

    This function processes a PyTree of parameters, replacing `LoraWeight.w`
    values and parameters marked with `LORA_FREEZE` in the `lora_spec` with
    `EmptyNode` instances. This is particularly useful for checkpointing,
    allowing the saving of only trainable parameters.

    Args:
        params: A PyTree of model parameters.
        lora_spec: A PyTree mirroring the structure of `params`, with values
            indicating how each parameter should be handled (LORA_FREEZE,
            LORA_FULL, or a LoRA dimension).

    Returns:
        A new PyTree with the specified parameters replaced by `EmptyNode`.
    """

    def node_mapper(node: Any, spec_val: Any) -> Any:
        if not isinstance(node, LoraWeight):
            return node if spec_val != LORA_FREEZE else EmptyNode
        children, aux = node.tree_flatten_with_keys()
        idx = next(i for i, (key, _) in enumerate(children) if key == "w")
        children[idx] = ("w", EmptyNode)
        return LoraWeight.tree_unflatten(aux, [c for _, c in children])

    return tree_map_with_implicit(node_mapper, params, lora_spec)


@dataclass
class RaptureConfig:
    """Configuration for the RaptureConfig fine-tuning method.

    Attributes:
        lora_dim: Dimensionality of LoRA adapters.
        fully_fine_tune_parameters: Optional list of parameter names to be fully fine-tuned.
        lora_fine_tune_parameters: Optional list of parameter names to be fine-tuned with LoRA.
        tune_vectors: Whether to tune vector parameters (length 1).
        verbose: If True, logs information about parameter tuning decisions.
        dtype: The dtype of the LoRA parameters.
    """

    lora_dim: int
    fully_fine_tune_parameters: Optional[List[str]] = None
    lora_fine_tune_parameters: Optional[List[str]] = None
    tune_vectors: bool = True
    verbose: bool = True
    dtype: jnp.dtype = jnp.float32


@dataclass
class RaptureModule:
    """Container for data related to a Flax module adapted with RaptureConfig.

    Attributes:
        lora_opt_state: The optimizer state for the LoRA parameters.
        lora_parameters: The PyTree of LoRA parameters.
        lora_module: The original Flax module adapted for use with LoRA parameters.
        lora_specs: A PyTree mirroring the parameter structure, indicating the tuning strategy for each parameter.
        lora_tx: The Optax gradient transformation wrapped for LoRA.
    """

    lora_opt_state: Union[jax.tree_util.PyTreeDef, dict]
    lora_parameters: Union[jax.tree_util.PyTreeDef, dict]
    lora_module: Union[Any, flax.linen.Module]
    lora_specs: Union[jax.tree_util.PyTreeDef, dict]
    lora_tx: optax.GradientTransformation


class LoraRapture:
    """Implements the RaptureConfig fine-tuning method.

    RaptureConfig combines LoRA (Low-Rank Adaptation) with full fine-tuning,
    allowing for efficient adaptation of large language models.
    """

    def __init__(self, config: RaptureConfig):
        """Initializes RaptureConfig with the provided configuration.

        Args:
            config: An `RaptureConfig` object specifying the fine-tuning setup.
        """
        self.config = config

    @staticmethod
    def merge_parameters(
        lora_parameters: Union[dict, jax.tree_util.PyTreeDef],
        destructive: bool = False,
    ) -> Union[dict, jax.tree_util.PyTreeDef]:
        """Merges LoRA parameters into the base model parameters.

        This function iterates through the PyTree of LoRA parameters, merging
        them with the corresponding base parameters within `LoraWeight`
        instances.  Optionally, it can destructively modify the input
        `lora_parameters` to free memory.

        Args:
            lora_parameters: A PyTree of LoRA parameters.
            destructive: If True, the input `lora_parameters` may be
                destructively modified to free memory.

        Returns:
            A new PyTree with LoRA parameters merged into the base parameters.
        """

        def _ensure_delete(val: jnp.ndarray) -> None:
            if not isinstance(val, jax.Array) or val.is_deleted():
                return
            try:
                val.device_buffer.delete()  # type: ignore
            except (ValueError, AttributeError):
                val.addressable_data(0).delete()

        materialize = jax.jit(
            materialize_nested, donate_argnums=0 if destructive else ()
        )

        def map_fn(param: Any) -> Any:
            if isinstance(param, LoraWeight):
                result = materialize(param)
                if destructive:
                    jax.tree_util.tree_map(_ensure_delete, param)
                return result
            return param

        return tree_map_with_implicit(map_fn, lora_parameters)

    def get_lora_parameters(self, lora_parameters) -> Dict:
        """
        Extract LoRA (Low-Rank Adaptation) parameters from a parameter tree.

        This method traverses the given parameter tree and extracts LoRA-specific
        parameters (a, b, and alpha) from LoraWeight instances. It flattens the
        resulting structure, removes any None values, and then unflattens it back
        into a nested dictionary.

        Args:
            lora_parameters (Any): A nested structure containing model parameters,
                                potentially including LoraWeight instances.

        Returns:
            Dict: A nested dictionary containing only the LoRA parameters. Each LoRA
                parameter set is represented as a dict with keys 'a', 'b', and 'alpha'.

        Note:
            - This method uses `tree_map_with_implicit` to traverse the parameter tree.
            - Parameters that are not instances of LoraWeight are mapped to None
            and subsequently removed from the result.
        """

        def map_fn(param: Any) -> Dict[str, Any] | None:
            """
            Extract LoRA parameters from a single parameter.

            Args:
                param (Any): A single parameter from the parameter tree.

            Returns:
                Dict[str, Any] | None: A dictionary containing LoRA parameters
                                    if the input is a LoraWeight instance,
                                    otherwise None.
            """
            if isinstance(param, LoraWeight):
                return {"a": param.a, "b": param.b, "alpha": param.alpha}
            return None

        pure_map = tree_map_with_implicit(map_fn, lora_parameters)
        pure_map = flax.traverse_util.flatten_dict(pure_map)
        keys = list(pure_map.keys())
        for key in keys:
            if pure_map[key] is None:
                pure_map.pop(key)
        return flax.traverse_util.unflatten_dict(pure_map)

    def base_decision_function(
        self,
        path: List[jax.tree_util.DictKey],
        params: Optional[Union[dict, jax.tree_util.PyTreeDef]] = None,
    ) -> int:
        """Determines the fine-tuning strategy for a parameter based on its path.

        This function acts as the default decision function for RaptureConfig. It
        checks if a parameter's path matches any entry in
        `fully_fine_tune_parameters` or `lora_fine_tune_parameters` from
        the `RaptureConfig`.  If a match is found, it returns `LORA_FULL`
        or the configured `lora_dim` respectively.  Otherwise, it defaults to
        `LORA_FREEZE`, indicating the parameter should not be fine-tuned.

        Args:
            path: A list of `jax.tree_util.DictKey` representing the path to
                the parameter within the model's parameter PyTree.
            params: This argument is unused and can be omitted.

        Returns:
            An integer representing the fine-tuning strategy for the parameter:
                * `LORA_FULL`: Full fine-tuning.
                * `LORA_FREEZE`: Freeze the parameter (no fine-tuning).
                * `self.config.lora_dim`: Fine-tune using LoRA with the
                    configured dimensionality.
        """
        del params  # Unused

        if self.config.fully_fine_tune_parameters is not None:
            for param_name in self.config.fully_fine_tune_parameters:
                if jax.tree_util.DictKey(key=param_name) in path:
                    if self.config.verbose:
                        logger.info(
                            f"Array {'/'.join(str(n.key) for n in path)} "
                            f"selected for full fine-tuning."
                        )
                    return LORA_FULL

        if self.config.lora_fine_tune_parameters is not None:
            for param_name in self.config.lora_fine_tune_parameters:
                if jax.tree_util.DictKey(key=param_name) in path:
                    if self.config.verbose:
                        logger.info(
                            f"Array {'/'.join(str(n.key) for n in path)} "
                            f"converted to LoraWeight with {self.config.lora_dim} dimensions."
                        )
                    return self.config.lora_dim

        return LORA_FREEZE

    def make_lora_specs(
        self,
        parameters: Union[dict, flax.core.FrozenDict],
        decision_fn: Optional[Callable] = None,
        tune_vectors: bool = False,
    ) -> Union[dict, jax.tree_util.PyTreeDef]:
        """Creates a PyTree specifying the fine-tuning strategy for parameters.

        This function generates a PyTree matching the structure of the input
        `parameters`.  Each value in the output tree represents the
        fine-tuning strategy for the corresponding parameter in `parameters`,
        determined by the provided `decision_fn`.

        Args:
            parameters: A PyTree of model parameters.
            decision_fn: A callable that takes the parameter path and the
                parameter array as input and returns an integer representing the
                fine-tuning strategy (`LORA_FREEZE`, `LORA_FULL`, or a
                LoRA dimension).  If None, `self.base_decision_function` is used.
            tune_vectors: Whether to tune vector parameters. If True, vectors
                will be fully fine-tuned.

        Returns:
            A PyTree mirroring the structure of `parameters`, with values
            indicating the fine-tuning strategy for each parameter.
        """
        decision_fn = (
            decision_fn if decision_fn is not None else self.base_decision_function
        )

        if decision_fn is None:

            def decision_fn(*args):  # pylint: disable=function-redefined
                return LORA_FREEZE

        def full_fn(path: List[jax.tree_util.DictKey], arr: jnp.ndarray) -> int:
            if len(arr.shape) < 2:
                return LORA_FULL if tune_vectors else LORA_FREEZE
            return decision_fn(path, arr)

        return jax.tree_util.tree_map_with_path(full_fn, parameters, is_leaf=None)

    @staticmethod
    def init_lora_parameters(
        param_tree: Union[dict, jax.tree_util.PyTreeDef],
        lora_spec: Union[dict, jax.tree_util.PyTreeDef],
        dtype: jnp.dtype = jnp.float32,
        rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
        stddev: float = 0.01,
        alpha: float = 1.0,
        is_leaf: bool = None,
    ) -> Union[dict, jax.tree_util.PyTreeDef]:
        """Initializes LoRA parameters based on the provided specifications.

        This function traverses the `param_tree`, initializing LoRA parameters
        according to the `lora_spec`.  Parameters marked for full
        fine-tuning (`LORA_FULL`) or freezing (`LORA_FREEZE`) retain their
        original form.  Parameters designated for LoRA tuning are replaced
        with `LoraWeight` instances, with adapter matrices initialized
        using a normal distribution.

        Args:
            param_tree: A PyTree of model parameters.
            lora_spec: A PyTree mirroring the structure of `param_tree`,
                specifying the fine-tuning strategy for each parameter.
            dtype: The data type for the LoRA parameters.
            rng: A JAX PRNG key for random initialization.
            stddev: Standard deviation for the normal distribution used to
                initialize LoRA adapter matrices.
            alpha: An additional scaling factor for the LoRA adapters.
            is_leaf: Optional function to determine if a node in the PyTree
                is a leaf.

        Returns:
            A PyTree matching the structure of `param_tree` with LoRA
            parameters initialized according to `lora_spec`.

        Raises:
            ValueError: If a vector parameter is encountered with a
                LoRA dimension specification (neither frozen nor fully tuned).
        """

        def iter_keys(key: jax.random.PRNGKey) -> jax.random.PRNGKey:
            while True:
                key, out_key = jax.random.split(key)
                yield out_key

        key_it = iter_keys(rng)

        def get_param(
            path: List[jax.tree_util.DictKey], param: jnp.ndarray, spec_val: int
        ) -> Any:
            if spec_val in (LORA_FREEZE, LORA_FULL):
                return param

            if len(param.shape) == 1:
                raise ValueError(
                    f"Vectors must either be frozen or fully tuned, but got "
                    f"lora_spec value {lora_spec} for param with path {path}"
                )

            if len(param.shape) == 2:
                b_dim, a_dim = param.shape
                b = jnp.zeros((b_dim, spec_val), dtype=dtype)
                a = (
                    jax.random.normal(next(key_it), (spec_val, a_dim), dtype=dtype)
                    * stddev
                )
                return LoraWeight(w=param, a=a, b=b, alpha=alpha)

            # Convolutional parameter case
            *window_shape, in_channels, out_channels = param.shape
            a = jnp.zeros(
                (*(1 for _ in range(len(window_shape))), spec_val, out_channels),
                dtype=param.dtype,
            )
            b = (
                jax.random.normal(
                    rng, (*window_shape, in_channels, spec_val), dtype=param.dtype
                )
                * stddev
            )
            return LoraWeight(param, a, b, alpha=alpha)

        return jax.tree_util.tree_map_with_path(
            get_param,
            param_tree,
            lora_spec,
            is_leaf=is_leaf,
        )

    @staticmethod
    def wrap_tx(
        tx: optax.GradientTransformation,
        lora_spec: Union[dict, jax.tree_util.PyTreeDef],
        scalar_frozen_grads: bool = False,
    ) -> optax.GradientTransformation:
        """Wraps an optimizer to handle LoRA parameters and freezing.

        This function customizes an Optax gradient transformation
        (`optax.GradientTransformation`) to work seamlessly with LoRA and
        parameter freezing.  It first creates a mask based on `lora_spec`,
        freezing parameters marked with `LORA_FREEZE`. Then it freezes the 'w'
        attribute of all `LoraWeight` instances, preventing the original
        weights from being updated.

        Args:
            tx: The base Optax gradient transformation to wrap.
            lora_spec: A PyTree specifying the fine-tuning strategy for each
                parameter.
            scalar_frozen_grads: If True, frozen gradients will be replaced
                with scalar zeros instead of zero-filled arrays. This can
                improve memory efficiency.

        Returns:
            The wrapped Optax gradient transformation that handles LoRA
            parameters and parameter freezing.
        """
        full_freeze_labels = jax.tree_util.tree_map(
            lambda x: "freeze" if x == LORA_FREEZE else "train",
            lora_spec,
        )
        optimizer_with_full_freeze = freeze_subtrees(
            tx,
            full_freeze_labels,
            use_scalar_zeros=scalar_frozen_grads,
        )
        return freeze_keys(
            optimizer_with_full_freeze,
            LoraWeight,
            "w",
            use_scalar_zeros=scalar_frozen_grads,
        )

    def apply_lora(
        self,
        module: Union[Any, flax.linen.Module],
        parameters: Union[dict, flax.core.FrozenDict],
        tx: optax.GradientTransformation,
        decision_fn: Optional[Callable] = None,
        tune_vectors: bool = False,
        rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
        stddev: float = 0.01,
        alpha: float = 1.0,
        is_leaf: bool = None,
        func_target: Optional[str] = None,
    ) -> RaptureModule:
        """Applies LoRA to a Flax module and wraps the optimizer.

        This function applies LoRA to the provided Flax module based on the
        `RaptureConfig` and the given `decision_fn`. It initializes LoRA
        parameters, wraps the optimizer to handle LoRA and frozen parameters,
        and returns an `XRaptureModule` containing the adapted module and
        related data.

        Args:
            module: The Flax module to apply LoRA to.
            parameters: The PyTree of parameters from the Flax module.
            tx: The Optax gradient transformation (optimizer) to use.
            decision_fn: An optional callable to override the default
                `base_decision_function` for determining parameter-specific
                fine-tuning strategies.
            tune_vectors: Whether to tune vector parameters using full fine-tuning.
            rng: A JAX PRNG key for random initialization.
            stddev: Standard deviation for initializing LoRA adapter matrices.
            alpha: Scaling factor for the LoRA adapters.
            is_leaf: Optional function to determine leaf nodes in PyTrees.

        Returns:
            An `XRaptureModule` containing the LoRA-adapted module, optimizer
            state, parameters, and specifications.
        """
        lora_spec = self.make_lora_specs(
            parameters=parameters,
            tune_vectors=tune_vectors,
            decision_fn=decision_fn,
        )

        lora_parameters = self.init_lora_parameters(
            param_tree=parameters,
            dtype=self.config.dtype,
            lora_spec=lora_spec,
            rng=rng,
            stddev=stddev,
            alpha=alpha,
            is_leaf=is_leaf,
        )
        tx = self.wrap_tx(tx=tx, lora_spec=lora_spec)
        opt_state = tx.init(lora_parameters)
        if func_target is None:
            lora_model = implicit_compact(module)
        else:
            lora_model = implicit_compact(getattr(module, func_target))
        return RaptureModule(
            lora_opt_state=opt_state,
            lora_module=lora_model,
            lora_specs=lora_spec,
            lora_parameters=lora_parameters,
            lora_tx=tx,
        )
