import flax.core
import jax.tree_util
import optax
import termcolor
from jax import numpy as jnp

from functools import partial
import warnings

from .implicit_array import (
    ImplicitArray,
    use_implicit_args,
    tree_map_with_implicit,
    default_handler,
    primitive_handler,
    ArrayValue,
    aux_field,
    EmptyNode,
    materialize_nested,
    freeze_keys,
    freeze_subtrees,
)

import jax

LORA_FREEZE = 0
LORA_FULL = -1

from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass
class LoraWeight(ImplicitArray):
    w: ArrayValue  # M x N
    a: ArrayValue  # k x N
    b: ArrayValue  # M x k

    alpha: float = aux_field(default=1.)

    def __post_init__(self):
        super().__post_init__()
        assert self.a.shape[-2] == self.b.shape[-1]
        assert self.w.shape[-2] == self.b.shape[-2]
        assert self.w.shape[-1] == self.a.shape[-1]

    def materialize(self):
        return (self.w + self.get_scale() * self.b @ self.a).astype(self.w.dtype)

    def get_scale(self):
        return self.alpha / self.b.shape[1]


def _check_dot_dimension_numbers(dimension_numbers):
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    if lhs_batch or rhs_batch:
        warnings.warn("Lorax does not support batched matmuls")
        return False
    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        warnings.warn("Lorax only supports matmul")
        return False
    return True


@primitive_handler("dot_general")
def handle_dot_lhs(
        primitive,
        lora: LoraWeight,
        rhs: ArrayValue,
        *,
        dimension_numbers,
        **kwargs
):
    if not _check_dot_dimension_numbers(dimension_numbers):
        return NotImplemented

    if isinstance(rhs, LoraWeight):
        rhs = rhs.materialize()
        warnings.warn("Encountered product of two LoraWeights. Materializing the rhs")

    op = partial(jax.lax.dot_general, **kwargs)

    lhs_contract, = dimension_numbers[0][0]

    first, second = (lora.a, lora.b) if lhs_contract == 1 else (lora.b, lora.a)

    first *= lora.get_scale()

    orig = op(lora.w, rhs, dimension_numbers=dimension_numbers)
    lora_product = op(first, rhs, dimension_numbers=dimension_numbers)

    second_dimension_numbers = ((lhs_contract,), (0,)), dimension_numbers[1]

    lora_product = op(second, lora_product, dimension_numbers=second_dimension_numbers)

    return (orig + lora_product).astype(orig.dtype)


@primitive_handler("dot_general")
def handle_dot_rhs(
        primitive,
        lhs: jax.Array,
        lora: LoraWeight,
        *,
        dimension_numbers,
        **kwargs
):
    if not _check_dot_dimension_numbers(dimension_numbers):
        return NotImplemented
    op = partial(jax.lax.dot_general, **kwargs)

    rhs_contract, = dimension_numbers[0][1]
    first, second = (lora.a, lora.b) if rhs_contract == 1 else (lora.b, lora.a)

    first *= lora.get_scale()

    orig = op(lhs, lora.w, dimension_numbers=dimension_numbers)
    lora_product = op(lhs, first, dimension_numbers=dimension_numbers)

    second_dimension_numbers = ((lhs.ndim - 1), (rhs_contract,)), dimension_numbers[1]

    lora_product = op(lora_product, second, dimension_numbers=second_dimension_numbers)

    return (orig + lora_product).astype(orig.dtype)


@primitive_handler("conv_general_dilated")
def handle_conv(
        primitive,
        inp: ArrayValue,
        lora: LoraWeight,
        *,
        dimension_numbers,
        **params
):
    if isinstance(inp, LoraWeight):
        warnings.warn("Using a LoraWeight as input to a convolution is not supported, so it will be materialized.")
        inp = inp.materialize()

    if not dimension_numbers.rhs_spec[:1] != (
            len(dimension_numbers.rhs_spec) - 1,
            len(dimension_numbers.rhs_spec) - 2,
    ):
        raise ValueError("Lorax only supports convolutions with shape (..., in_features, out_features)")

    params = {**params, "dimension_numbers": dimension_numbers}
    op = partial(jax.lax.conv_general_dilated, **params)
    orig = op(inp, lora.w)

    lora_product = op(inp, lora.b)

    params["window_strides"] = (1,) * (len(dimension_numbers.rhs_spec) - 2)
    params["padding"] = "VALID"
    lora_product = jax.lax.conv_general_dilated(
        lora_product,
        lora.a * lora.get_scale(),
        **params
    )

    return (orig + lora_product).astype(orig.dtype)


@primitive_handler("gather")
def handle_gather(
        primitive,
        lora: LoraWeight,
        indices: jax.Array,
        *,
        dimension_numbers,
        slice_sizes,
        **params
):
    if dimension_numbers.offset_dims != (len(indices.shape) - 1,):
        return NotImplemented

    lora_dim = lora.b.shape[-1]

    if slice_sizes != (1, lora.a.shape[1]):
        return NotImplemented

    params = {**params, "dimension_numbers": dimension_numbers}

    orig = jax.lax.gather(lora.w, indices, slice_sizes=slice_sizes, **params)

    new_slice_sizes = (1, lora_dim)

    lora_product = jax.lax.gather(lora.b, indices, slice_sizes=new_slice_sizes, **params)
    lora_product = lora_product @ (lora.a * lora.get_scale())

    return (orig + lora_product).astype(orig.dtype)


@primitive_handler("transpose")
def eval_lora_transpose(
        primitive,
        arg: LoraWeight,
        *,
        permutation
):
    if not len(arg.shape) == 2 and permutation == (1, 0):
        return NotImplemented

    return LoraWeight(
        w=arg.w.T,
        a=arg.b.T,
        b=arg.a.T,
        alpha=arg.alpha,
    )


@primitive_handler("convert_element_type")
def eval_lora_convert_element_type(primitive, arg: LoraWeight, **params):
    result = jax.tree_map(
        partial(default_handler, primitive, **params),
        arg
    )
    result.dtype = params["new_dtype"]
    return result


def split_lora_params(params, lora_spec):
    """
    Map params to a pytree in which all `LoraWeight.w` values and all params marked with
    LORA_FREEZE are replaced with EmptyNode. This is useful for checkpointing just
    the trainable params.
    """

    def node_mapper(node, spec_val):
        if not isinstance(node, LoraWeight):
            return node if spec_val != LORA_FREEZE else EmptyNode
        children, aux = node.tree_flatten_with_keys()
        idx = next(i for i, (key, _) in enumerate(children) if key == "w")
        children[idx] = ("w", EmptyNode)

        return LoraWeight.tree_unflatten(aux, [c for _, c in children])

    return tree_map_with_implicit(node_mapper, params, lora_spec)


@dataclass
class XRapTureConfig:
    lora_dim: int
    fully_fine_tune_parameters: Optional[list[str]] = None
    lora_fine_tune_parameters: Optional[list[str]] = None
    tune_vectors: bool = True
    verbose: bool = True
    dtype: jnp.dtype = jnp.float32


@dataclass
class XRapTureModule:
    lora_opt_state: jax.tree_util.PyTreeDef | dict
    lora_parameters: jax.tree_util.PyTreeDef | dict
    lora_module: Any | flax.linen.Module
    lora_specs: jax.tree_util.PyTreeDef | dict
    lora_tx: optax.GradientTransformation


class XRapTure:
    def __init__(
            self,
            config: XRapTureConfig
    ):
        self.config = config

    @staticmethod
    def merge_parameters(
            lora_parameters,
            destructive=True,
    ):
        def _ensure_delete(val):
            if not isinstance(val, jax.Array) or val.is_deleted():
                return
            val.device_buffer.delete()

        materialize = jax.jit(materialize_nested, donate_argnums=0 if destructive else ())

        def map_fn(param):
            if isinstance(param, LoraWeight):
                result = materialize(param)
                if destructive:
                    jax.tree_map(_ensure_delete, param)
                return result
            return param

        return tree_map_with_implicit(map_fn, lora_parameters)

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
                                f" {'/'.join(str(n.key) for n in path)} "
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
                                f" {'/'.join(str(n.key) for n in path)} "
                                f"Selected for LoRA Fine-Tune with {self.config.lora_dim} dimensions.",
                                color="cyan",
                                force_color=True
                            )
                        )
                    return self.config.lora_dim

        return LORA_FREEZE

    def make_lora_specs(
            self,
            parameters: dict | flax.core.FrozenDict,
            decision_fn: Optional[Callable] = None,
            tune_vectors: bool = False,
    ):
        decision_fn = decision_fn if decision_fn is not None else self.base_decision_function

        if decision_fn is None:
            def decision_fn(*args):
                return LORA_FREEZE

        def full_fn(path, arr):
            if len(arr.shape) < 2:
                return LORA_FULL if tune_vectors else LORA_FREEZE

            value = decision_fn(path, arr)
            return value

        return jax.tree_util.tree_map_with_path(full_fn, parameters, is_leaf=None)

    @staticmethod
    def init_lora_parameters(
            param_tree,
            lora_spec,
            dtype: jnp.dtype = jnp.float32,
            rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
            stddev: float = 0.01,
            alpha: float = 1.,
            is_leaf: bool = None
    ):

        def iter_keys(key):
            while True:
                key, out_key = jax.random.split(key)
                yield out_key

        key_it = iter_keys(rng)

        def get_param(path, param, spec_val):
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
                a = jax.random.normal(next(key_it), (spec_val, a_dim), dtype=dtype) * stddev
                return LoraWeight(w=param, a=a, b=b, alpha=alpha)

            # conv case
            *window_shape, in_channels, out_channels = param.shape

            a = jnp.zeros((
                *(1 for _ in range(len(window_shape))),
                spec_val,
                out_channels
            ), dtype=param.dtype)
            b = jax.random.normal(
                rng, (
                    *window_shape,
                    in_channels,
                    spec_val
                ), dtype=param.dtype
            ) * stddev
            return LoraWeight(
                param,
                a,
                b,
                alpha=alpha
            )

        return jax.tree_util.tree_map_with_path(
            get_param,
            param_tree,
            lora_spec,
            is_leaf=is_leaf
        )

    @staticmethod
    def wrap_tx(
            tx: optax.GradientTransformation,
            lora_spec,
            scalar_frozen_grads=False
    ):
        full_freeze_labels = jax.tree_map(
            lambda x: "freeze" if x == LORA_FREEZE else "train",
            lora_spec
        )
        optimizer_with_full_freeze = freeze_subtrees(
            tx,
            full_freeze_labels,
            use_scalar_zeros=scalar_frozen_grads
        )

        return freeze_keys(
            optimizer_with_full_freeze,
            LoraWeight,
            "w",
            use_scalar_zeros=scalar_frozen_grads
        )

    def apply_lora(
            self,
            module: Any | flax.linen.Module,
            parameters: dict | flax.core.FrozenDict,
            tx: optax.GradientTransformation,
            decision_fn: Optional[Callable] = None,
            tune_vectors: bool = False,
            rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
            stddev: float = 0.01,
            alpha: float = 1.,
            is_leaf: bool = None
    ) -> XRapTureModule:
        lora_spec = self.make_lora_specs(
            parameters=parameters,
            tune_vectors=tune_vectors,
            decision_fn=decision_fn
        )

        lora_parameters = self.init_lora_parameters(
            param_tree=parameters,
            dtype=self.config.dtype,
            lora_spec=lora_spec,
            rng=rng,
            stddev=stddev,
            alpha=alpha,
            is_leaf=is_leaf
        )
        tx = self.wrap_tx(
            tx=tx,
            lora_spec=lora_spec
        )
        opt_state = tx.init(lora_parameters)
        lora_model = use_implicit_args(
            module
        )

        return XRapTureModule(
            lora_opt_state=opt_state,
            lora_module=lora_model,
            lora_specs=lora_spec,
            lora_parameters=lora_parameters,
            lora_tx=tx
        )
