# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# duplicated in order to add quantization parameters

"""Linear modules."""

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Mapping,
    Dict,
    TypeVar,
    Hashable,
    Protocol, Generic
)
import re
import chex
import flax.traverse_util
import jax.tree_util
from flax.core import FrozenDict
from flax.linen.dtypes import promote_dtype
from flax.linen.module import compact
from flax.linen.module import Module
import flax.struct
from jax import eval_shape
from jax.core import ShapedArray
import numpy as np
import dataclasses
import functools
import jax
import jax.numpy as jnp
from jax import lax
from flax.linen import initializers

from flax.linen import dtypes, module, transforms

# General

Array = Union[jax.Array, Any]
PRNGKey = jax.Array
RNGSequences = Dict[str, PRNGKey]
Dtype = Union[jax.typing.DTypeLike, Any]
Shape = Sequence[int]
K = TypeVar('K')


class Key(Hashable, Protocol):
    def __lt__(self: K, value: K, /) -> bool:
        ...


Path = str
PathParts = Tuple[Key, ...]

Leaf = Any

# Linear

PrecisionLike = Union[
    None,
    str,
    jax.lax.Precision,
    Tuple[str, str],
    Tuple[jax.lax.Precision, jax.lax.Precision],
]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]

PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]

# Initializers

Initializer = Union[jax.nn.initializers.Initializer, Callable[..., Any]]

# Collections

Collection = Mapping[str, Any]
MutableCollection = Dict[str, Any]

# Dicts

VariableDict = Mapping[str, Collection]
FrozenVariableDict = FrozenDict[str, Collection]
MutableVariableDict = Dict[str, MutableCollection]

PRNGFoldable = Union[int, str]

# Axes

T = TypeVar('T')


@dataclasses.dataclass(frozen=True)
class In(Generic[T]):
    """Specifies a variable collection should only be lifted as input."""

    axis: T


@dataclasses.dataclass(frozen=True)
class Out(Generic[T]):
    """Specifies a variable collection should only be lifted as output."""

    axis: T


Axis = Optional[int]
InOutAxis = Union[Axis, In[Axis], Out[Axis]]

ScanAxis = int
InOutScanAxis = Union[ScanAxis, In[ScanAxis], Out[ScanAxis]]

Axes = Union[int, Sequence[int]]

# SPMD

LogicalNames = Tuple[Union[str, None], ...]

# Maps each logical axis  to physical mesh, can be either None (replicated),
# one physical axis or a tuple of physical axes.
LogicalRules = Sequence[Tuple[str, Union[str, Tuple[str, ...], None]]]
ArrayPytree = Any  # pylint: disable=invalid-name
LogicalPartitionSpec = Any  # pylint: disable=invalid-name
LogicalPartitionSpecPytree = Any  # pylint: disable=invalid-name
PartitionSpecPytree = Any  # pylint: disable=invalid-name

Sharding = Tuple[Optional[str], ...]
field = dataclasses.field
canonicalize_dtype = dtypes.canonicalize_dtype
merge_param = module.merge_param
map_variables = transforms.map_variables

default_kernel_init = initializers.lecun_normal()


def quantize(
        array: jnp.ndarray,
        int_dtype: jnp.dtype = jnp.int8,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    scale = jnp.max(jnp.abs(array), axis=-1, keepdims=True)
    array = jax.lax.convert_element_type(
        jnp.rint(array * ((jnp.iinfo(int_dtype).max + abs(jnp.iinfo(int_dtype).min)) / 2 / scale)), int_dtype
    )
    return array, scale


def de_quantize(
        quantized: jnp.ndarray,
        scale: jnp.ndarray,
        float_dtype: jnp.dtype = jnp.float16,
        threshold: float = 1e-6
):
    max_scale = (jnp.iinfo(quantized.dtype).max + abs(jnp.iinfo(quantized.dtype).min)) / 2
    return ((jax.lax.convert_element_type(quantized, float_dtype) * scale) / max_scale) + threshold


@flax.struct.dataclass
class LinearBitKernel:
    kernel: Array
    scale: Array

    @property
    def shape(self):
        return self.kernel.shape

    @property
    def dtype(self):
        return self.kernel.dtype

    @property
    def sharding(self):
        return self.kernel.sharding

    @property
    def size(self):
        return self.kernel.size


# jax.tree_util.register_pytree_node(
#     LinearBitKernel,
#     lambda x: ([x.kernel, x.scale], ()),
#     lambda _, children: LinearBitKernel(children[0], children[1])
# )

def quantize_parameters(
        filter_list_quantization: list,
        params: dict,
        int_dtype: jnp.dtype = jnp.int8
):
    pattern = re.compile("({})".format("|".join(filter_list_quantization)))

    def lam_func(path, array):
        if pattern.search("/".join(p.key for p in path)):
            return LinearBitKernel(
                *quantize(array, int_dtype=int_dtype)
            )
        return array

    return jax.tree_util.tree_map_with_path(lam_func, params)


def de_quantize_params(
        params: jax.tree_util.PyTreeDef,
        dtype: jnp.dtype = jnp.float32,
        shard_funcs: Optional[Mapping[str, Callable[[chex.Array], chex.Array]]] = None
):
    def _q(pr):
        if isinstance(pr, LinearBitKernel):
            return jnp.array(
                de_quantize(
                    pr.kernel, pr.scale, dtype, 0
                )
            )
        return pr

    prm = flax.traverse_util.flatten_dict(params)
    for key in list(prm.keys()):
        value = _q(prm[key])
        if shard_funcs is not None:
            value = shard_funcs[key](value)
        prm[key] = value
    return flax.traverse_util.unflatten_dict(prm)


def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


def _canonicalize_tuple(x: Union[Sequence[int], int]) -> Tuple[int, ...]:
    if isinstance(x, Iterable):
        return tuple(x)
    else:
        return (x,)


def control_quantization(array, param_dtype):
    if isinstance(array, LinearBitKernel):
        array = de_quantize(
            array.kernel,
            array.scale,
            param_dtype,
            .0
        )
    return array


class Linear(Module):
    """A linear transformation applied over the last dimension of the input.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    """

    features: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    # Deprecated. Will be removed.
    dot_general: Optional[DotGeneralT] = None
    dot_general_cls: Any = None

    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )

        kernel = control_quantization(kernel, self.param_dtype)

        if self.use_bias:
            bias = self.param(
                "bias",
                self.bias_init,
                (self.features,),
                self.param_dtype
            )
            bias = control_quantization(bias, self.param_dtype)
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        elif self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = lax.dot_general
        y = dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]


def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
    """ "Canonicalizes conv padding to a jax.lax supported format."""
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)] * rank
    if isinstance(padding, Sequence) and len(padding) == rank:
        new_pad = []
        for p in padding:
            if isinstance(p, int):
                new_pad.append((p, p))
            elif isinstance(p, tuple) and len(p) == 2:
                new_pad.append(p)
            else:
                break
        if len(new_pad) == rank:
            return new_pad
    raise ValueError(
        f"Invalid padding format: {padding}, should be str, int,"
        f" or a sequence of len {rank} where each element is an"
        " int or pair of ints."
    )


class _Conv(Module):
    """Convolution Module wrapping `lax.conv_general_dilated[_local]`.

    Attributes:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: an integer or a sequence of `n` integers, representing the
        inter-window strides (default: 1).
      padding: either the string `"SAME"`, the string `"VALID"`, the string
        `"CIRCULAR"` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. A single int is interpreted as applying the same padding
        in all dims and assign a single int in a sequence causes the same padding
        to be used on both sides. `"CAUSAL"` padding for a 1D convolution will
        left-pad the convolution axis, resulting in same-sized output.
      input_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as "atrous convolution".
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      use_bias: whether to add a bias to the output (default: True).
      mask: Optional mask for the weights during masked convolution. The mask must
            be the same shape as the convolution weight matrix.
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    """

    features: int
    kernel_size: Sequence[int]
    strides: Union[None, int, Sequence[int]] = 1
    padding: PaddingLike = "SAME"
    input_dilation: Union[None, int, Sequence[int]] = 1
    kernel_dilation: Union[None, int, Sequence[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    mask: Optional[Array] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        initializers.zeros_init()
    )
    # Deprecated. Will be removed.
    conv_general_dilated: Optional[ConvGeneralDilatedT] = None
    conv_general_dilated_cls: Any = None

    @property
    def shared_weights(self) -> bool:  # type: ignore
        """Defines whether weights are shared or not between different pixels.

        Returns:
          `True` to use shared weights in convolution (regular convolution).
          `False` to use different weights at different pixels, a.k.a.
          "locally connected layer", "unshared convolution", or "local convolution".

        """
        ...

    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a (potentially unshared) convolution to the inputs.

        Args:
          inputs: input data with dimensions (*batch_dims, spatial_dims...,
            features). This is the channels-last convention, i.e. NHWC for a 2d
            convolution and NDHWC for a 3D convolution. Note: this is different from
            the input convention used by `lax.conv_general_dilated`, which puts the
            spatial dimensions last.
            Note: If the input has more than 1 batch dimension, all batch dimensions
            are flattened into a single dimension for the convolution and restored
            before returning.  In some cases directly vmap"ing the layer may yield
            better performance than this default flattening approach.  If the input
            lacks a batch dimension it will be added for the convolution and removed
            n return, an allowance made to enable writing single-example code.

        Returns:
          The convolved data.
        """

        if isinstance(self.kernel_size, int):
            raise TypeError(
                "Expected Conv kernel_size to be a"
                " tuple/list of integers (eg.: [3, 3]) but got"
                f" {self.kernel_size}."
            )
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(
                x: Optional[Union[int, Sequence[int]]]
        ) -> Tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        # Combine all input batch dimensions into a single leading batch axis.
        num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (total_batch_size,) + inputs.shape[
                                                     num_batch_dimensions:
                                                     ]
            inputs = jnp.reshape(inputs, flat_input_shape)

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax = canonicalize_padding(self.padding, len(kernel_size))
        if padding_lax == "CIRCULAR":
            kernel_size_dilated = [
                (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
            ]
            zero_pad: List[Tuple[int, int]] = [(0, 0)]
            pads = (
                    zero_pad
                    + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
                    + [(0, 0)]
            )
            inputs = jnp.pad(inputs, pads, mode="wrap")
            padding_lax = "VALID"
        elif padding_lax == "CAUSAL":
            if len(kernel_size) != 1:
                raise ValueError(
                    "Causal padding is only implemented for 1D convolutions."
                )
            left_pad = kernel_dilation[0] * (kernel_size[0] - 1)
            pads = [(0, 0), (left_pad, 0), (0, 0)]
            inputs = jnp.pad(inputs, pads)
            padding_lax = "VALID"

        dimension_numbers = _conv_dimension_numbers(inputs.shape)
        in_features = jnp.shape(inputs)[-1]

        if self.shared_weights:
            # One shared convolutional kernel for all pixels in the output.
            assert in_features % self.feature_group_count == 0
            kernel_shape = kernel_size + (
                in_features // self.feature_group_count,
                self.features,
            )

        else:
            if self.feature_group_count != 1:
                raise NotImplementedError(
                    "`lax.conv_general_dilated_local` does not support "
                    f"`feature_group_count != 1`, got `{self.feature_group_count}`."
                )

            # Need to know the spatial output shape of a standard convolution to
            # create the unshared convolution kernel.
            if self.conv_general_dilated_cls is not None:
                conv_general_dilated = self.conv_general_dilated_cls()
            elif self.conv_general_dilated is not None:
                conv_general_dilated = self.conv_general_dilated
            else:
                conv_general_dilated = lax.conv_general_dilated
            conv_output_shape = eval_shape(
                lambda lhs, rhs: conv_general_dilated(  # pylint: disable=g-long-lambda
                    lhs=lhs,
                    rhs=rhs,
                    window_strides=strides,
                    padding=padding_lax,
                    dimension_numbers=dimension_numbers,
                    lhs_dilation=input_dilation,
                    rhs_dilation=kernel_dilation,
                ),
                inputs,
                ShapedArray(kernel_size + (in_features, self.features), inputs.dtype),
            ).shape

            # One (unshared) convolutional kernel per each pixel in the output.
            kernel_shape = conv_output_shape[1:-1] + (
                np.prod(kernel_size) * in_features,
                self.features,
            )

        if self.mask is not None and self.mask.shape != kernel_shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. "
                f"Shapes are: {self.mask.shape}, {kernel_shape}"
            )

        kernel = self.param(
            "kernel", self.kernel_init, kernel_shape, self.param_dtype
        )
        kernel = control_quantization(kernel, self.param_dtype)
        if self.mask is not None:
            kernel *= self.mask

        if self.use_bias:
            if self.shared_weights:
                # One bias weight per output channel, shared between pixels.
                bias_shape = (self.features,)
            else:
                # One bias weight per output entry, unshared betwen pixels.
                bias_shape = conv_output_shape[1:]  # type: ignore

            bias = self.param("bias", self.bias_init, bias_shape, self.param_dtype)
            bias = control_quantization(bias, self.param_dtype)
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        if self.shared_weights:
            if self.conv_general_dilated_cls is not None:
                conv_general_dilated = self.conv_general_dilated_cls()
            elif self.conv_general_dilated is not None:
                conv_general_dilated = self.conv_general_dilated
            else:
                conv_general_dilated = lax.conv_general_dilated
            y = conv_general_dilated(
                inputs,
                kernel,
                strides,
                padding_lax,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                feature_group_count=self.feature_group_count,
                precision=self.precision,
            )
        else:
            y = lax.conv_general_dilated_local(
                lhs=inputs,
                rhs=kernel,
                window_strides=strides,
                padding=padding_lax,
                filter_shape=kernel_size,
                lhs_dilation=input_dilation,
                rhs_dilation=kernel_dilation,
                dimension_numbers=dimension_numbers,
                precision=self.precision,
            )

        if self.use_bias:
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y += bias

        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)
        return y


class Conv(_Conv):
    """Convolution Module wrapping `lax.conv_general_dilated`.

    Attributes:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: an integer or a sequence of `n` integers, representing the
        inter-window strides (default: 1).
      padding: either the string `"SAME"`, the string `"VALID"`, the string
        `"CIRCULAR"` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. A single int is interpreted as applying the same padding
        in all dims and assign a single int in a sequence causes the same padding
        to be used on both sides. `"CAUSAL"` padding for a 1D convolution will
        left-pad the convolution axis, resulting in same-sized output.
      input_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as "atrous convolution".
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      use_bias: whether to add a bias to the output (default: True).
      mask: Optional mask for the weights during masked convolution. The mask must
            be the same shape as the convolution weight matrix.
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    """

    @property
    def shared_weights(self) -> bool:
        return True


class ConvLocal(_Conv):
    """Local convolution Module wrapping `lax.conv_general_dilated_local`.

    Attributes:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: an integer or a sequence of `n` integers, representing the
        inter-window strides (default: 1).
      padding: either the string `"SAME"`, the string `"VALID"`, the string
        `"CIRCULAR"` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. A single int is interpreted as applying the same padding
        in all dims and assign a single int in a sequence causes the same padding
        to be used on both sides. `"CAUSAL"` padding for a 1D convolution will
        left-pad the convolution axis, resulting in same-sized output.
      input_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`
        (default: 1). Convolution with input dilation `d` is equivalent to
        transposed convolution with stride `d`.
      kernel_dilation: an integer or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel (default: 1). Convolution with kernel dilation
        is also known as "atrous convolution".
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      use_bias: whether to add a bias to the output (default: True).
      mask: Optional mask for the weights during masked convolution. The mask must
            be the same shape as the convolution weight matrix.
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    """

    @property
    def shared_weights(self) -> bool:
        return False


class ConvTranspose(Module):
    """Convolution Module wrapping lax.conv_transpose.

    Attributes:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel. For 1D convolution,
        the kernel size can be passed as an integer. For all other cases, it must
        be a sequence of integers.
      strides: a sequence of `n` integers, representing the inter-window strides.
      padding: either the string `"SAME"`, the string `"VALID"`, the string
        `"CIRCULAR"` (periodic boundary conditions), or a sequence of `n` `(low,
        high)` integer pairs that give the padding to apply before and after each
        spatial dimension. A single int is interpreted as applying the same padding
        in all dims and assign a single int in a sequence causes the same padding
        to be used on both sides.
      kernel_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel. Convolution with kernel dilation is also known as "atrous
        convolution".
      use_bias: whether to add a bias to the output (default: True).
      mask: Optional mask for the weights during masked convolution. The mask must
            be the same shape as the convolution weight matrix.
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
      transpose_kernel: if True flips spatial axes and swaps the input/output
        channel axes of the kernel.
    """

    features: int
    kernel_size: Union[int, Sequence[int]]
    strides: Optional[Sequence[int]] = None
    padding: PaddingLike = "SAME"
    kernel_dilation: Optional[Sequence[int]] = None
    use_bias: bool = True
    mask: Optional[Array] = None
    dtype: Dtype = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        initializers.zeros_init()
    )
    transpose_kernel: bool = False

    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a transposed convolution to the inputs.

        Behaviour mirrors of `jax.lax.conv_transpose`.

        Args:
          inputs: input data with dimensions (*batch_dims, spatial_dims...,
            features). This is the channels-last convention, i.e. NHWC for a 2d
            convolution and NDHWC for a 3D convolution. Note: this is different from
            the input convention used by `lax.conv_general_dilated`, which puts the
            spatial dimensions last.
            Note: If the input has more than 1 batch dimension, all batch dimensions
            are flattened into a single dimension for the convolution and restored
            before returning.  In some cases directly vmap"ing the layer may yield
            better performance than this default flattening approach.  If the input
            lacks a batch dimension it will be added for the convolution and removed
            n return, an allowance made to enable writing single-example code.

        Returns:
          The convolved data.
        """
        kernel_size: Tuple[int, ...]
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size,)
        else:
            kernel_size = tuple(self.kernel_size)

        # Combine all input batch dimensions into a single leading batch axis.
        num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (total_batch_size,) + inputs.shape[
                                                     num_batch_dimensions:
                                                     ]
            inputs = jnp.reshape(inputs, flat_input_shape)

        strides: Tuple[int, ...]
        if self.strides is None:
            strides = (1,) * (inputs.ndim - 2)
        else:
            strides = tuple(self.strides)

        in_features = jnp.shape(inputs)[-1]
        if self.transpose_kernel:
            kernel_shape = kernel_size + (self.features, in_features)
        else:
            kernel_shape = kernel_size + (in_features, self.features)

        if self.mask is not None and self.mask.shape != kernel_shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. "
                f"Shapes are: {self.mask.shape}, {kernel_shape}"
            )

        kernel = self.param(
            "kernel", self.kernel_init, kernel_shape, self.param_dtype
        )
        kernel = control_quantization(kernel, self.param_dtype)
        if self.mask is not None:
            kernel *= self.mask

        padding_lax = canonicalize_padding(self.padding, len(kernel_size))
        if padding_lax == "CIRCULAR":
            padding_lax = "VALID"

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
            bias = control_quantization(bias, self.param_dtype)
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        y = lax.conv_transpose(
            inputs,
            kernel,
            strides,
            padding_lax,
            rhs_dilation=self.kernel_dilation,
            transpose_kernel=self.transpose_kernel,
            precision=self.precision,
        )

        if self.padding == "CIRCULAR":
            # For circular padding, we need to identify the size of the final output
            # ("period") along each spatial dimension, pad each dimension to an
            # integer number of periods, and wrap the array periodically around each
            # dimension. Padding should be done in such a way that the start of the
            # original input data inside the padded array is located at integer
            # number of periods - otherwise the result would be circularly shifted.

            # Compute period along each spatial dimension - it"s input size scaled
            # by the stride.
            scaled_x_dims = [
                x_dim * stride
                for x_dim, stride in zip(jnp.shape(inputs)[1:-1], strides)
            ]
            # Compute difference between the current size of y and the final output
            # size, and complement this difference to 2 * period - that gives how
            # much we need to pad.
            size_diffs = [
                -(y_dim - x_dim) % (2 * x_dim)
                for y_dim, x_dim in zip(y.shape[1:-1], scaled_x_dims)
            ]
            if self.transpose_kernel:
                # If the kernel is transposed, the "+1" is put on the right to
                # mirror the regular convolution. If the same kernel parameters are used
                # as for Conv, this layer then computes the proper transpose convolution.
                total_pad = [
                    (size_diff // 2, (size_diff + 1) // 2) for size_diff in size_diffs
                ]
            else:
                # Divide the padding equally between left and right. The choice to put
                # "+1" on the left (and not on the right) represents a convention for
                # aligning even-sized kernels.
                total_pad = [
                    ((size_diff + 1) // 2, size_diff // 2) for size_diff in size_diffs
                ]
            y = jnp.pad(y, [(0, 0)] + total_pad + [(0, 0)])
            # Wrap the result periodically around each spatial dimension,
            # one by one.
            for i in range(1, y.ndim - 1):
                y = y.reshape(
                    y.shape[:i] + (-1, scaled_x_dims[i - 1]) + y.shape[i + 1:]
                )
                y = y.sum(axis=i)

        if self.use_bias:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)

        return y


default_embed_init = initializers.variance_scaling(
    1.0, "fan_in", "normal", out_axis=0
)


class Embed(Module):
    """Embedding Module.

    A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
      num_embeddings: number of embeddings.
      features: number of feature dimensions for each embedding.
      dtype: the dtype of the embedding vectors (default: same as embedding).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      embedding_init: embedding initializer.
    """

    num_embeddings: int
    features: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    embedding_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_embed_init

    embedding: Array = dataclasses.field(init=False)

    def setup(self):
        self.embedding = self.param(
            "embedding",
            self.embedding_init,
            (self.num_embeddings, self.features),
            self.param_dtype,
        )

    def __call__(self, inputs: Array) -> Array:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        (embedding,) = promote_dtype(
            control_quantization(self.embedding, self.param_dtype), dtype=self.dtype, inexact=False
        )
        return jnp.take(embedding, inputs, axis=0)

    def attend(self, query: Array) -> Array:
        """Attend over the embedding using a query array.

        Args:
          query: array with last dimension equal the feature depth `features` of the
            embedding.
        Returns:
          An array with final dim `num_embeddings` corresponding to the batched
          inner-product of the array of query vectors against each embedding.
          Commonly used for weight-sharing between embeddings and logit transform
          in NLP models.
        """
        query, embedding = promote_dtype(query, control_quantization(self.embedding, self.param_dtype),
                                         dtype=self.dtype)
        return jnp.dot(query, embedding.T)


def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, Iterable):
        axes = (axes,)
    return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _abs_sq(x):
    """Computes the elementwise square of the absolute value |x|^2."""
    if jnp.iscomplexobj(x):
        return lax.square(lax.real(x)) + lax.square(lax.imag(x))
    else:
        return lax.square(x)


def _compute_stats(
        x: Array,
        axes: Axes,
        dtype: Optional[Dtype],
        axis_name: Optional[str] = None,
        axis_index_groups: Any = None,
        use_mean: bool = True,
        use_fast_variance: bool = True,
        mask: Optional[Array] = None,
):
    """Computes mean and variance statistics.
  
    This implementation takes care of a few important details:
    - Computes in float32 precision for stability in half precision training.
    - If `use_fast_variance` is `True`, mean and variance are computed using
      Var = E[|x|^2] - |E[x]|^2, instead of Var = E[|x - E[x]|^2]), in a single
      XLA fusion.
    - Clips negative variances to zero which can happen due to
      roundoff errors. This avoids downstream NaNs.
    - Supports averaging across a parallel axis and subgroups of a parallel axis
      with a single `lax.pmean` call to avoid latency.
  
    Arguments:
      x: Input array.
      axes: The axes in ``x`` to compute mean and variance statistics for.
      dtype: Optional dtype specifying the minimal precision. Statistics are
        always at least float32 for stability (default: dtype of x).
      axis_name: Optional name for the pmapped axis to compute mean over. Note,
        this is only used for pmap and shard map. For SPMD jit, you do not need to
        manually synchronize. Just make sure that the axes are correctly annotated
        and XLA:SPMD will insert the necessary collectives.
      axis_index_groups: Optional axis indices.
      use_mean: If true, calculate the mean from the input and use it when
        computing the variance. If false, set the mean to zero and compute the
        variance without subtracting the mean.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
      mask: Binary array of shape broadcastable to `inputs` tensor, indicating
        the positions for which the mean and variance should be computed.
  
    Returns:
      A pair ``(mean, var)``.
    """
    if dtype is None:
        dtype = jnp.result_type(x)
    # promote x to at least float32, this avoids half precision computation
    # but preserves double or complex floating points
    dtype = jnp.promote_types(dtype, jnp.float32)
    x = jnp.asarray(x, dtype)
    axes = _canonicalize_axes(x.ndim, axes)

    def maybe_distributed_mean(*xs, mask=None):
        mus = tuple(x.mean(axes, where=mask) for x in xs)
        if axis_name is None:
            return mus if len(xs) > 1 else mus[0]
        else:
            # In the distributed case we stack multiple arrays to speed comms.
            if len(xs) > 1:
                reduced_mus = lax.pmean(
                    jnp.stack(mus, axis=0),
                    axis_name,
                    axis_index_groups=axis_index_groups,
                )
                return tuple(reduced_mus[i] for i in range(len(xs)))
            else:
                return lax.pmean(mus[0], axis_name, axis_index_groups=axis_index_groups)

    if use_mean:
        if use_fast_variance:
            mu, mu2 = maybe_distributed_mean(x, _abs_sq(x), mask=mask)
            # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
            # to floating point round-off errors.
            var = jnp.maximum(0.0, mu2 - _abs_sq(mu))
        else:
            mu = maybe_distributed_mean(x, mask=mask)
            var = maybe_distributed_mean(
                _abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask
            )
    else:
        var = maybe_distributed_mean(_abs_sq(x), mask=mask)
        mu = jnp.zeros_like(var)
    return mu, var


def _normalize(
        mdl: Module,
        x: Array,
        mean: Array,
        var: Array,
        reduction_axes: Axes,
        feature_axes: Axes,
        dtype: Optional[Dtype],
        param_dtype: Dtype,
        epsilon: float,
        use_bias: bool,
        use_scale: bool,
        bias_init: Initializer,
        scale_init: Initializer,
):
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])

    mean = jnp.expand_dims(mean, reduction_axes)
    var = jnp.expand_dims(var, reduction_axes)
    y = x - mean
    mul = lax.rsqrt(var + epsilon)
    args = [x]
    if use_scale:
        scale = control_quantization(mdl.param(
            "scale", scale_init, reduced_feature_shape, param_dtype
        ), param_dtype).reshape(feature_shape)
        mul *= scale
        args.append(scale)
    y *= mul
    if use_bias:
        bias = control_quantization(mdl.param(
            "bias", bias_init, reduced_feature_shape, param_dtype
        ), param_dtype).reshape(feature_shape)
        y += bias
        args.append(bias)
    dtype = dtypes.canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, dtype)


def _l2_normalize(x, axis=None, eps=1e-12):
    """Normalizes along dimension `axis` using an L2 norm.
  
    This specialized function exists for numerical stability reasons.
  
    Args:
      x: An input ndarray.
      axis: Dimension along which to normalize, e.g. `1` to separately normalize
        vectors in a batch. Passing `None` views `t` as a flattened vector when
        calculating the norm (equivalent to Frobenius norm).
      eps: Epsilon to avoid dividing by zero.
  
    Returns:
      An array of the same shape as "x" L2-normalized along "axis".
    """
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class BatchNorm(Module):
    """BatchNorm Module.
  
    Usage Note:
    If we define a model with BatchNorm, for example::
  
      >>> import flax.linen as nn
      >>> import jax, jax.numpy as jnp
      >>> BN = nn.BatchNorm(momentum=0.9, epsilon=1e-5, dtype=jnp.float32)
  
    The initialized variables dict will contain, in addition to a "params"
    collection, a separate "batch_stats" collection that will contain all the
    running statistics for all the BatchNorm layers in a model::
  
      >>> x = jax.random.normal(jax.random.key(0), (5, 6))
      >>> variables = BN.init(jax.random.key(1), x, use_running_average=False)
      >>> jax.tree_util.tree_map(jnp.shape, variables)
      {"batch_stats": {"mean": (6,), "var": (6,)}, "params": {"bias": (6,), "scale": (6,)}}
  
    We then update the batch_stats during training by specifying that the
    ``batch_stats`` collection is mutable in the ``apply`` method for our
    module.::
  
      >>> y, new_batch_stats = BN.apply(variables, x, mutable=["batch_stats"], use_running_average=False)
  
    During eval we would define BN with ``use_running_average=True`` and use the
    batch_stats collection from training to set the statistics.  In this case
    we are not mutating the batch statistics collection, and needn"t mark it
    mutable::
  
      >>> y = BN.apply(variables, x, mutable=["batch_stats"], use_running_average=True)
  
    Attributes:
      use_running_average: if True, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of the batch
        statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  if True, bias (beta) is added.
      use_scale: if True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See ``jax.pmap`` for a description of axis names (default: None).
        Note, this is only used for pmap and shard map. For SPMD jit, you do not
        need to manually synchronize. Just make sure that the axes are correctly
        annotated and XLA:SPMD will insert the necessary collectives.
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
        examples on the first two and last two devices. See ``jax.lax.psum`` for
        more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Initializer = initializers.zeros
    scale_init: Initializer = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(
            self,
            x,
            use_running_average: Optional[bool] = None,
            *,
            mask: Optional[jax.Array] = None,
    ):
        """Normalizes the input using batch statistics.
    
        NOTE:
        During initialization (when ``self.is_initializing()`` is ``True``) the running
        average of the batch statistics will not be updated. Therefore, the inputs
        fed during initialization don"t need to match that of the actual input
        distribution and the reduction axis (set with ``axis_name``) does not have
        to exist.
    
        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.
          mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.
    
        Returns:
          Normalized inputs (the same shape as inputs).
        """

        use_running_average = module.merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = control_quantization(self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        ), jnp.float32)
        ra_var = control_quantization(self.variable(
            "batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape
        ), jnp.float32)

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
                mask=mask,
            )

            if not self.is_initializing():
                ra_mean.value = (
                        self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        return _normalize(
            self,
            x,
            mean,
            var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


class LayerNorm(Module):
    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Initializer = initializers.zeros
    scale_init: Initializer = initializers.ones
    reduction_axes: Axes = -1
    feature_axes: Axes = -1
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, *, mask: Optional[jax.Array] = None):
        """Applies layer normalization on the input.
    
        Args:
          x: the inputs
          mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.
    
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        mean, var = _compute_stats(
            x,
            self.reduction_axes,
            self.dtype,
            self.axis_name,
            self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )

        return _normalize(
            self,
            x,
            mean,
            var,
            self.reduction_axes,
            self.feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


class RMSNorm(Module):
    """RMS Layer normalization (https://arxiv.org/abs/1910.07467).
  
    RMSNorm normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    Unlike LayerNorm which re-centers the mean to be 0 and normalizes by the
    standard deviation of the activations, RMSNorm does not re-center at all
    and instead normalizes by the root mean square of the activations.
  
    Example usage::
  
      >>> import flax.linen as nn
      >>> import jax
  
      >>> x = jax.random.normal(jax.random.key(0), (5, 6))
      >>> layer = nn.RMSNorm()
      >>> variables = layer.init(jax.random.key(1), x)
      >>> variables
      {"params": {"scale": Array([1., 1., 1., 1., 1., 1.], dtype=float32)}}
      >>> y = layer.apply(variables, x)
  
    Attributes:
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      scale_init: Initializer for scale, by default, one.
      reduction_axes: Axes for computing normalization statistics.
      feature_axes: Feature axes for learned bias and scaling.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See ``jax.pmap`` for a description of axis names (default: None).
        This is only needed if the model is subdivided across devices, i.e. the
        array being normalized is sharded across devices within a pmap or shard
        map. For SPMD jit, you do not need to manually synchronize. Just make sure
        that the axes are correctly annotated and XLA:SPMD will insert the
        necessary collectives.
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
        examples on the first two and last two devices. See ``jax.lax.psum`` for
        more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_scale: bool = True
    scale_init: Initializer = initializers.ones
    reduction_axes: Axes = -1
    feature_axes: Axes = -1
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, *, mask: Optional[jax.Array] = None):
        """Applies RMS layer normalization on the input.
    
        Args:
          x: the inputs
          mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.
    
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        mean, var = _compute_stats(
            x,
            self.reduction_axes,
            self.dtype,
            self.axis_name,
            self.axis_index_groups,
            use_mean=False,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )

        return _normalize(
            self,
            x,
            mean,
            var,
            self.reduction_axes,
            self.feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            False,
            self.use_scale,
            initializers.zeros,
            self.scale_init,
        )


class GroupNorm(Module):
    num_groups: Optional[int] = 32
    group_size: Optional[int] = None
    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Initializer = initializers.zeros
    scale_init: Initializer = initializers.ones
    reduction_axes: Optional[Axes] = None
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, *, mask: Optional[jax.Array] = None):
        """Applies group normalization to the input (arxiv.org/abs/1803.08494).
    
        Args:
          x: the input of shape ``...C`` where ``C`` is a channels dimension and ``...``
            represents an arbitrary number of extra dimensions that can be used to
            accumulate statistics over. If no reduction axes have been specified
            then all additional dimensions ``...`` will be used to accumulate
            statistics apart from the leading dimension which is assumed to
            represent the batch.
          mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.
    
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        if self.reduction_axes is not None:
            reduction_axes = self.reduction_axes
        else:
            reduction_axes = list(range(1, x.ndim - 1)) + [-1]
        feature_axis = -1

        reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)

        if reduction_axes[-1] != (feature_axis % x.ndim):
            raise ValueError(
                "The reduction axes must include the final dimension "
                "as this is assumed to be the feature axis."
            )

        if (self.num_groups is None and self.group_size is None) or (
                self.num_groups is not None and self.group_size is not None
        ):
            raise ValueError(
                "Either `num_groups` or `group_size` should be "
                "specified. If `group_size` is to be specified, "
                "pass `num_groups=None` as argument to override "
                "the default `num_groups` value of 32."
            )

        channels = x.shape[-1]
        if self.group_size is not None:
            if channels % self.group_size != 0:
                raise ValueError(
                    "Number of channels ({}) is not multiple of the "
                    "group size ({}).".format(channels, self.group_size)
                )
            num_groups = channels // self.group_size
        else:
            num_groups = self.num_groups
            assert isinstance(num_groups, int)

        if num_groups <= 0 or channels % num_groups != 0:
            raise ValueError(
                "Number of groups ({}) does not divide the number"
                " of channels ({}).".format(num_groups, channels)
            )

        group_size = x.shape[-1] // num_groups
        group_shape = x.shape[:-1] + (num_groups, group_size)

        if mask is not None:
            mask = mask.reshape(mask.shape[:-1] + (num_groups, group_size))

        mean, var = _compute_stats(
            x.reshape(group_shape),
            list(reduction_axes[:-1]) + [-1],
            self.dtype,
            self.axis_name,
            self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )
        mean = jnp.repeat(mean, group_size, axis=-1)
        var = jnp.repeat(var, group_size, axis=-1)

        return _normalize(
            self,
            x,
            mean,
            var,
            reduction_axes[:-1],
            (feature_axis,),
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


class InstanceNorm(Module):
    """Instance normalization (https://arxiv.org/abs/1607.08022v3).
  
    InstanceNorm normalizes the activations of the layer for each channel (rather
    than across all channels like Layer Normalization), and for each given example
    in a batch independently (rather than across an entire batch like Batch
    Normalization). i.e. applies a transformation that maintains the mean activation
    within each channel within each example close to 0 and the activation standard
    deviation close to 1.
  
    NOTE: This normalization operation is identical to LayerNorm and GroupNorm; the
    difference is simply which axes are reduced and the shape of the feature axes
    (i.e. the shape of the learnable scale and bias parameters).
  
    Example usage::
  
      >>> import flax.linen as nn
      >>> import jax
      >>> import numpy as np
  
      >>> # dimensions: (batch, height, width, channel)
      >>> x = jax.random.normal(jax.random.key(0), (2, 3, 4, 5))
      >>> layer = nn.InstanceNorm()
      >>> variables = layer.init(jax.random.key(1), x)
      >>> variables
      {"params": {"scale": Array([1., 1., 1., 1., 1.], dtype=float32), "bias": Array([0., 0., 0., 0., 0.], dtype=float32)}}
      >>> y = layer.apply(variables, x)
  
      >>> # having a channel_axis of -1 in InstanceNorm is identical to reducing all non-batch,
      >>> # non-channel axes and using the feature_axes as the feature_axes in LayerNorm
      >>> y2 = nn.LayerNorm(reduction_axes=[1, 2], feature_axes=-1).apply(variables, x)
      >>> np.testing.assert_allclose(y, y2, atol=1e-7)
      >>> y3 = nn.GroupNorm(num_groups=x.shape[-1]).apply(variables, x)
      >>> np.testing.assert_allclose(y, y3, atol=1e-7)
  
    Attributes:
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  If True, bias (beta) is added.
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
      feature_axes: Axes for features. The learned bias and scaling parameters will
        be in the shape defined by the feature axes. All other axes except the batch
        axes (which is assumed to be the leading axis) will be reduced.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See ``jax.pmap`` for a description of axis names (default: None).
        This is only needed if the model is subdivided across devices, i.e. the
        array being normalized is sharded across devices within a pmap or shard
        map. For SPMD jit, you do not need to manually synchronize. Just make sure
        that the axes are correctly annotated and XLA:SPMD will insert the
        necessary collectives.
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
        examples on the first two and last two devices. See ``jax.lax.psum`` for
        more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Initializer = initializers.zeros
    scale_init: Initializer = initializers.ones
    feature_axes: Axes = -1
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, *, mask: Optional[jax.Array] = None):
        """Applies instance normalization on the input.
    
        Args:
          x: the inputs
          mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.
    
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        feature_axes = _canonicalize_axes(x.ndim, self.feature_axes)
        if 0 in feature_axes:
            raise ValueError("The channel axes cannot include the leading dimension "
                             "as this is assumed to be the batch axis.")
        reduction_axes = [i for i in range(1, x.ndim) if i not in feature_axes]

        mean, var = _compute_stats(
            x,
            reduction_axes,
            self.dtype,
            self.axis_name,
            self.axis_index_groups,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )

        return _normalize(
            self,
            x,
            mean,
            var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


class SpectralNorm(Module):
    layer_instance: Module
    n_steps: int = 1
    epsilon: float = 1e-12
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    error_on_non_matrix: bool = False
    collection_name: str = "batch_stats"

    @compact
    def __call__(self, *args, update_stats: bool, **kwargs):
        """Compute the largest singular value of the weights in ``self.layer_instance``
        using power iteration and normalize the weights using this value before
        computing the ``__call__`` output.
    
        Args:
          *args: positional arguments to be passed into the call method of the
            underlying layer instance in ``self.layer_instance``.
          update_stats: if True, update the internal ``u`` vector and ``sigma``
            value after computing their updated values using power iteration. This
            will help the power iteration method approximate the true singular value
            more accurately over time.
          **kwargs: keyword arguments to be passed into the call method of the
            underlying layer instance in ``self.layer_instance``.
    
        Returns:
          Output of the layer using spectral normalized weights.
        """

        def layer_forward(layer_instance):
            return layer_instance(*args, **kwargs)

        return transforms.map_variables(
            layer_forward,
            trans_in_fn=lambda vs: jax.tree_util.tree_map_with_path(
                functools.partial(
                    self._spectral_normalize,
                    update_stats=update_stats,
                ),
                vs,
            ),
            init=self.is_initializing(),
            mutable=True,
        )(self.layer_instance)

    def _spectral_normalize(self, path, vs, update_stats):
        value = jnp.asarray(vs)
        value_shape = value.shape

        # Skip and return value if input is scalar, vector or if number of power
        # iterations is less than 1
        if value.ndim <= 1 or self.n_steps < 1:
            return value
        # Handle higher-order tensors.
        elif value.ndim > 2:
            if self.error_on_non_matrix:
                raise ValueError(
                    f"Input is {value.ndim}D but error_on_non_matrix is True"
                )
            else:
                value = jnp.reshape(value, (-1, value.shape[-1]))

        u_var_name = (
                self.layer_instance.name
                + "/"
                + "/".join((dict_key.key for dict_key in path[1:]))
                + "/u"
        )
        u_var = control_quantization(self.variable(
            self.collection_name,
            u_var_name,
            jax.random.normal,
            self.make_rng("params")
            if not self.has_variable(self.collection_name, u_var_name)
            else None,
            (1, value.shape[-1]),
            self.param_dtype,
        ), self.param_dtype)
        u0 = u_var.value
        sigma_var_name = (
                self.layer_instance.name
                + "/"
                + "/".join((dict_key.key for dict_key in path[1:]))
                + "/sigma"
        )
        sigma_var = control_quantization(self.variable(
            self.collection_name, sigma_var_name, jnp.ones, (), self.param_dtype
        ), self.param_dtype)

        # Power iteration for the weight"s singular value.
        for _ in range(self.n_steps):
            v0 = _l2_normalize(
                jnp.matmul(u0, value.transpose([1, 0])), eps=self.epsilon
            )
            u0 = _l2_normalize(jnp.matmul(v0, value), eps=self.epsilon)

        u0 = jax.lax.stop_gradient(u0)
        v0 = jax.lax.stop_gradient(v0)

        sigma = jnp.matmul(jnp.matmul(v0, value), jnp.transpose(u0))[0, 0]

        value /= jnp.where(sigma != 0, sigma, 1)
        value_bar = value.reshape(value_shape)

        if update_stats:
            u_var.value = u0
            sigma_var.value = sigma

        dtype = dtypes.canonicalize_dtype(vs, u0, v0, sigma, dtype=self.dtype)
        return jnp.asarray(value_bar, dtype)


class WeightNorm(Module):
    layer_instance: Module
    epsilon: float = 1e-12
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_scale: bool = True
    scale_init: Initializer = initializers.ones
    feature_axes: Optional[Axes] = -1
    variable_filter: Optional[Iterable] = dataclasses.field(
        default_factory=lambda: {"kernel"}
    )

    @compact
    def __call__(self, *args, **kwargs):
        """Compute the l2-norm of the weights in ``self.layer_instance``
        and normalize the weights using this value before computing the
        ``__call__`` output.
    
        Args:
          *args: positional arguments to be passed into the call method of the
            underlying layer instance in ``self.layer_instance``.
          **kwargs: keyword arguments to be passed into the call method of the
            underlying layer instance in ``self.layer_instance``.
    
        Returns:
          Output of the layer using l2-normalized weights.
        """

        def layer_forward(layer_instance):
            return layer_instance(*args, **kwargs)

        return transforms.map_variables(
            layer_forward,
            trans_in_fn=lambda vs: jax.tree_util.tree_map_with_path(
                self._l2_normalize,
                vs,
            ),
            init=self.is_initializing(),
        )(self.layer_instance)

    def _l2_normalize(self, path, vs):
        """Compute the l2-norm and normalize the variables ``vs`` using this
        value. This is intended to be a helper function used in this Module"s
        ``__call__`` method in conjunction with ``nn.transforms.map_variables``
        and ``jax.tree_util.tree_map_with_path``.
    
        Args:
          path: dict key path, used for naming the ``scale`` variable
          vs: variables to be l2-normalized
        """
        value = jnp.asarray(vs)
        str_path = (
                self.layer_instance.name
                + "/"
                + "/".join((dict_key.key for dict_key in path[1:]))
        )
        if self.variable_filter:
            for variable_name in self.variable_filter:
                if variable_name in str_path:
                    break
            else:
                return value

        if self.feature_axes is None:
            feature_axes = ()
            reduction_axes = tuple(i for i in range(value.ndim))
        else:
            feature_axes = _canonicalize_axes(value.ndim, self.feature_axes)
            reduction_axes = tuple(
                i for i in range(value.ndim) if i not in feature_axes
            )

        feature_shape = [1] * value.ndim
        reduced_feature_shape = []
        for ax in feature_axes:
            feature_shape[ax] = value.shape[ax]
            reduced_feature_shape.append(value.shape[ax])

        value_bar = _l2_normalize(value, axis=reduction_axes, eps=self.epsilon)

        args = [vs]
        if self.use_scale:
            scale = control_quantization(
                self.param(
                    str_path + "/scale",
                    self.scale_init,
                    reduced_feature_shape,
                    self.param_dtype,
                ), self.param_dtype
            ).reshape(feature_shape)
            value_bar *= scale
            args.append(scale)

        dtype = dtypes.canonicalize_dtype(*args, dtype=self.dtype)
        return jnp.asarray(value_bar, dtype)
