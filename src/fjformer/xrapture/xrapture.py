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
from typing import Optional, Callable, Any, Union


@dataclass
class LoraWeight(ImplicitArray):
    w: ArrayValue  # M x N
    a: ArrayValue  # k x N
    b: ArrayValue  # M x k

    alpha: float = aux_field(default=1.)

    def __post_init__(self):
        """
        The __post_init__ function is called after the __init__ function.
        It allows us to check that the shapes of our parameters are correct, and if not, raise an error.
        
        :param self: Represent the instance of the class
        :return: The output of the super()
        
        """
        super().__post_init__()
        assert self.a.shape[-2] == self.b.shape[-1], "A and B Array should be like a[K,N] B[M,K]"
        assert self.w.shape[-2] == self.b.shape[-2], "A and B Array should be like W[M,N] B[M,K]"
        assert self.w.shape[-1] == self.a.shape[-1], "A and B Array should be like W[M,N] A[K,N]"

    def materialize(self):
        """
        The materialize function is used to create a new matrix from the parameters of the factorization.
        
        :param self: Access the attributes and methods of a class
        :return: The materialized vector
        
        """
        return (self.w + self.get_scale() * self.b @ self.a).astype(self.w.dtype)

    def get_scale(self):
        """
        The get_scale function returns the scale of the model.
        The scale is defined as alpha / number of columns in b.
        
        
        :param self: Represent the instance of the class
        :return: The scale of the model
        
        """
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
    """
    The handle_dot_lhs function is a JAX primitive that allows us to perform
    matrix multiplication on LoraWeights. It does this by first performing the matrix
    multiplication on the underlying weight tensor, and then adding in a second term
    that accounts for the fact that we are multiplying two low-rank matrices together.
    
    :param primitive: Determine which function to use
    :param lora: LoraWeight: Pass the loraweight object to the function
    :param rhs: ArrayValue: Pass the right hand side of the dot product
    :param dimension_numbers: Determine which dimensions are being contracted
    :param kwargs: Pass the dimension_numbers to handle_dot_lhs
    :return: The result of the dot product between
    
    """
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
    """
    The handle_dot_rhs function is a partial application of the jax.lax.dot_general function,
    which takes in two arrays and returns their dot product (or matrix multiplication). The 
    handle_dot_rhs function is used to handle the case where a LoraWeight object appears on the right-hand side of an equation. 
    The handle_dot_rhs function takes in three arguments: primitive, lhs, and lora (the LoraWeight object). It then checks that 
    the dimension numbers are correct for this operation using _check_dimension numbers(). If they are not correct it will return NotIm
    
    :param primitive: Identify the function that is being called
    :param lhs: jax.Array: Store the left hand side of the dot product
    :param lora: LoraWeight: Pass the loraweight object to the function
    :param dimension_numbers: Specify the dimensions of the input arrays
    :param kwargs: Pass the dimension_numbers argument to handle_dot_rhs
    :return: The output of the dot product with a loraweight
    
    """
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
    """
    The handle_conv function is a helper function that allows us to use LoraWeight objects as inputs to convolutions.
    
    :param primitive: Identify the function that is being called
    :param inp: ArrayValue: Specify the input to the convolution
    :param lora: LoraWeight: Pass the loraweight object into the function
    :param *: Pass in the dimension_numbers parameter
    :param dimension_numbers: Specify the convolution
    :param params: Pass in the dimension_numbers parameter
    :return: The result of the convolution
    
    """
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
    """
    The handle_gather function is a JAX primitive handler that allows us to
    perform the gather operation on LoraWeight objects. This function is called by
    JAX when it encounters a gather operation in the computation graph. The function
    takes as input:
    
    :param primitive: Identify the operation
    :param lora: LoraWeight: Pass the loraweight object to the function
    :param indices: jax.Array: Select the rows of the weight matrix
    :param dimension_numbers: Specify the dimension numbers of
    :param slice_sizes: Specify the size of each slice
    :param params: Pass the dimension_numbers parameter to the gather function
    :return: A new loraweight
    
    """
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
    """
    The eval_lora_transpose function is used to transpose a LoraWeight object.
    
    :param primitive: Determine which function to use
    :param arg: LoraWeight: Specify the type of input that is expected
    :param *: Indicate that the permutation parameter is a keyword-only argument
    :param permutation: Specify the permutation of the weights
    :return: A loraweight object with the same
    
    """
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
    lora_opt_state: Union[jax.tree_util.PyTreeDef, dict]
    lora_parameters: Union[jax.tree_util.PyTreeDef, dict]
    lora_module: Union[Any , flax.linen.Module]
    lora_specs: Union[jax.tree_util.PyTreeDef, dict]
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

        """    
        The merge_parameters function is used to convert a LoraWeight into an array.
        
        :param lora_parameters: Pass in the parameters of the model
        :param destructive: Determine whether to delete the original parameters or not
        :param : Determine if the function is destructive or not
        :return: The parameters of the model
        
        """

        def _ensure_delete(val):
            if not isinstance(val, jax.Array) or val.is_deleted():
                return
            try:
                val.device_buffer.delete()
            except ValueError:
                val.device_buffers.delete()

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
            params: Optional[Union[dict, jax.tree_util.PyTreeDef]] = None
    ):

        """    
        The base_decision_function function is used to determine which parameters are frozen,
        which are fine-tuned with LoRA, and which are fully fine-tuned. The function takes in a path
        to the parameter (e.g., &quot;model/dense_layer/kernel&quot;) and returns an integer indicating how 
        the parameter should be treated:
        
        :param self: Refer to the object itself
        :param path: list[jax.tree_util.DictKey]: Determine the path of the parameter in question
        :param params: dict | jax.tree_util.PyTreeDef | None: Specify the parameters of the model
        :return: The following:
        
        """
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
            parameters: Union[dict, flax.core.FrozenDict],
            decision_fn: Optional[Callable] = None,
            tune_vectors: bool = False,
    ):

        """    
        The make_lora_specs function is used to create a dictionary of LORA specs for the parameters
        of a model. The function takes in two arguments:
        
        :param self: Allow the function to access other attributes and methods of the class
        :param parameters: dict | flax.core.FrozenDict: Specify the parameters to be tuned
        :param decision_fn: Optional[Callable]: Decide whether to freeze or unfreeze a parameter
        :param tune_vectors: bool: Determine if the vectors should be tuned or not
        :param : Decide whether to freeze the parameter or not
        :return: A dictionary of the same shape as the input parameters,
        
        """
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

        """
        The init_lora_parameters function takes in a parameter tree, the lora_spec, and some other parameters.
        It then iterates through the parameter tree using jax.tree_util.tree_map_with_path to get each path and value of 
        the parameter tree (which is just a nested dictionary). It then checks if that value is either LORA_FREEZE or
        LORA_FULL (these are constants defined above). If it's one of those two values, it returns the original parameter as-is; 
        otherwise it creates a new LoraWeight object with random values for b
        
        :param param_tree: Specify the parameters of a neural network
        :param lora_spec: Determine how many parameters to tune
        :param dtype: jnp.dtype: Specify the data type of the parameters
        :param rng: jax.random.PRNGKey: Generate random numbers
        :param stddev: float: Initialize the weights of the network
        :param alpha: float: Control the amount of regularization
        :param is_leaf: bool: Specify whether a node is a leaf or not
        :return: A tree of loraweight objects
        
        """

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

        """    
        The wrap_tx function takes a gradient transformation and wraps it in two
        freeze transformations. The first freezes all parameters that are marked as
        LORA_FREEZE, which is the default for LoraWeight objects. The second freezes
        the weights of all LoraWeight objects, regardless of their freeze status.
        
        :param tx: optax.GradientTransformation: Pass in the optimizer
        :param lora_spec: Specify which parameters we want to freeze
        :param scalar_frozen_grads: Determine whether to use scalar zeros or array zeros
        :return: A transformed version of the optimizer
        
        """
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
            module: Union[Any, flax.linen.Module],
            parameters: Union[dict, flax.core.FrozenDict],
            tx: optax.GradientTransformation,
            decision_fn: Optional[Callable] = None,
            tune_vectors: bool = False,
            rng: jax.random.PRNGKey = jax.random.PRNGKey(0),
            stddev: float = 0.01,
            alpha: float = 1.,
            is_leaf: bool = None
    ) -> XRapTureModule:

        """
        The apply_lora function is a wrapper for the XRapTureModule class.
        It takes in a module, parameters, and an optimizer (tx) and returns an instance of the XRapTureModule class.
        The apply_lora function also allows you to specify whether you want to tune vectors as well as
        whether you want to use a decision function when tuning your parameters. The default behavior is that
        vectors are tuned using LORA while scalars are tuned using SGD.

        :param self: Access the attributes of the class
        :param module: Any | flax.linen.Module: Specify the model that is being trained
        :param parameters: dict | flax.core.FrozenDict: Define the parameters of the model
        :param tx: optax.GradientTransformation: Specify the optimizer
        :param decision_fn: Optional[Callable]: Decide whether to apply lora to a parameter or not
        :param tune_vectors: bool: Determine whether to tune the vectors or not
        :param rng: jax.random.PRNGKey: Set the random seed for the initialisation of parameters
        :param stddev: float: Set the standard deviation of the initial weights
        :param alpha: float: Control the variance of the gaussian distribution used to initialize
        :param is_leaf: bool: Determine if the node is a leaf or not
        :return: A XRaptureModule object
        """
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
