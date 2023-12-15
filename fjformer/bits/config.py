# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration dataclasses."""

import dataclasses
from typing import Any, Callable, Optional
from . import calibration
from . import stochastic_rounding
from . import int_numerics
from . import no_numerics
from . import numerics
import jax
import jax.numpy as jnp

DType = Any
Context = Any  # TODO(lew): We could put Context in a separate file.

ClipAndRoundFn = Callable[[jnp.ndarray, Context], jnp.ndarray]


@dataclasses.dataclass(slots=True)
class Tensor:
    """Configuration of quantization of one tensor or one side of tensor op."""

    numerics: numerics.QNumerics
    calib_shared_axes: Optional[list[int]]
    scale_stop_grad: bool
    # noise+clip+round
    # We apply gradient of clip_and_round in bwd pass.
    calibration: calibration.Calibration
    # Round up the calibration to power of 2 (po2).
    po2_scale: bool
    use_fake_quant: bool
    # Controls at what value of input tensor should be used.
    # Setting it to True, but not quantizing fwd pass will assert-fail.
    use_fwd_quant: Optional[bool]
    # Operations for retrieving or storing quantized tensors and their scales
    # TODO(yichizh): Factor out auxilliary dataclasses into a separate file.
    # The following dtype Any should be q_dot_general.QTensor but that triggers
    # recursive importing
    preprocess: Optional[Callable[[Optional[Any]], Optional[Any]]]

    @classmethod
    def make(cls, *args, **kwargs) -> 'Tensor':
        return tensor_make(*args, **kwargs)


@dataclasses.dataclass(slots=True)
class LocalQ:
    contraction_axis_shard_count: int


@dataclasses.dataclass(slots=True)
class DotGeneralRaw:
    """Configuration of quantization of one dot_general without gradient."""

    lhs: Tensor
    rhs: Tensor
    dg_accumulator_dtype: Optional[DType]
    local_aqt: Optional[LocalQ]

    @classmethod
    def make(cls, *args, **kwargs) -> 'DotGeneralRaw':
        """
        The make function is a factory function that creates an instance of the DotGeneralRaw class.

        :param cls: Create a new instance of the class
        :param args: Send a non-keyworded variable length argument list to the function
        :param kwargs: Pass a variable number of keyword arguments to the function
        :return: A dotgeneralraw object
        """
        return dot_general_raw_make(*args, **kwargs)

    @classmethod
    def make_conv_general_dilated(cls, *args, **kwargs) -> 'DotGeneralRaw':
        """
        The make_conv_general_dilated function is a wrapper for the conv_general_dilated_make function.
        It allows us to use the make function in our DotGeneralRaw class, which we can then use as a
        component of our network. The make function takes in arguments that are used by the
        conv_general_dilated_make function to create an instance of DotGeneralRaw.

        :param cls: Create a new instance of the class
        :param args: Send a non-keyworded variable length argument list to the function
        :param kwargs: Pass a variable number of keyword arguments to a function
        :return: A dotgeneralraw object
        """
        return conv_general_dilated_make(*args, **kwargs)


@dataclasses.dataclass(slots=True)
class DotGeneral:
    """Configuration of quantization of dot_general and its gradients."""

    fwd: DotGeneralRaw
    dlhs: DotGeneralRaw
    drhs: DotGeneralRaw

    @classmethod
    def make(cls, *args, **kwargs) -> 'DotGeneral':
        return dot_general_make(*args, **kwargs)


def set_fwd_numerics(cfg, fwd_numerics: numerics.QNumerics):
    """
    The set_fwd_numerics function sets the numerics of the forward problem.

    :param cfg: Store the configuration of the simulation
    :param fwd_numerics: numerics.QNumerics: Set the numerical
    :return: The configuration object with the numerics for the forward problem set
    """
    cfg.fwd.lhs.numerics = fwd_numerics
    cfg.fwd.rhs.numerics = fwd_numerics


def set_accumulator_dtype(
        cfg: DotGeneral,
        fwd_dtype: Optional[DType],
        dlhs_dtype: Optional[DType],
        drhs_dtype: Optional[DType],
):
    """
    The set_accumulator_dtype function sets the accumulator dtype for each of the three
    differentiable functions.  The accumulator dtype is used to store intermediate results
    during forward and backward passes.  It is also used to store gradients during backward pass.
    The default value for this parameter is None, which means that it will be set automatically by
    the library based on other parameters such as input data types and output data type.

    :param cfg: DotGeneral: Set the accumulator dtype for all three
    :param fwd_dtype: Optional[DType]: Set the dtype of the forward pass
    :param dlhs_dtype: Optional[DType]: Set the data type of the left hand side
    :param drhs_dtype: Optional[DType]: Set the data type for the drhs accumulator
    :param : Set the dtype of the accumulator
    """
    cfg.fwd.dg_accumulator_dtype = fwd_dtype
    cfg.dlhs.dg_accumulator_dtype = dlhs_dtype
    cfg.drhs.dg_accumulator_dtype = drhs_dtype


def set_stochastic_rounding(
        cfg: DotGeneral,
        vjp_lhs_stochastic_rounding: bool,
        vjp_rhs_stochastic_rounding: bool,
        implementation: str,
):
    """Configure stochastic rounding implementation."""
    noise_implementations = {
        'jax.uniform': lambda shape, key: jax.random.uniform(key, shape) - 0.5,
        'custom-1': stochastic_rounding.random_centered_uniform,
    }
    msg = f'{implementation} not supported.'
    assert implementation in noise_implementations.keys(), msg
    noise_fn = noise_implementations[implementation]

    if vjp_lhs_stochastic_rounding:
        cfg.dlhs.lhs.numerics = cfg.dlhs.lhs.numerics.replace(noise_fn=noise_fn)
        cfg.drhs.lhs.numerics = cfg.drhs.lhs.numerics.replace(noise_fn=noise_fn)
    else:
        cfg.dlhs.lhs.numerics = cfg.dlhs.lhs.numerics.replace(noise_fn=None)
        cfg.drhs.lhs.numerics = cfg.drhs.lhs.numerics.replace(noise_fn=None)

    if vjp_rhs_stochastic_rounding:
        cfg.dlhs.rhs.numerics = cfg.dlhs.rhs.numerics.replace(noise_fn=noise_fn)
        cfg.drhs.rhs.numerics = cfg.drhs.rhs.numerics.replace(noise_fn=noise_fn)
    else:
        cfg.dlhs.rhs.numerics = cfg.dlhs.rhs.numerics.replace(noise_fn=None)
        cfg.drhs.rhs.numerics = cfg.drhs.rhs.numerics.replace(noise_fn=None)


def set_static_bound(cfg: DotGeneral, bound: float = 1.0):

    """
    The set_static_bound function sets the calibration of all the forward and backward
    differentiation operators to a constant value. This is useful for testing purposes, as it
    allows us to check that our implementation is correct by comparing against known values.

    :param cfg: DotGeneral: Set the bounds for each of the six functions in a dotgeneral object
    :param bound: float: Set the bound of the calibration
    """
    cfg.fwd.lhs.calibration = calibration.ConstantCalibration(bound)
    cfg.fwd.rhs.calibration = calibration.ConstantCalibration(bound)
    cfg.drhs.lhs.calibration = calibration.ConstantCalibration(bound)
    cfg.drhs.rhs.calibration = calibration.ConstantCalibration(bound)
    cfg.dlhs.lhs.calibration = calibration.ConstantCalibration(bound)
    cfg.dlhs.rhs.calibration = calibration.ConstantCalibration(bound)


def tensor_make(bits: Optional[int]) -> 'Tensor':

    """
    The tensor_make function is a helper function that creates a Tensor object.

    :param bits: Optional[int]: Set the number of bits for quantization
    :return: A tensor object
    """
    if bits is None:
        effective_numerics = no_numerics.NoNumerics()
    else:
        pz = False if bits == 1 else True
        dtype = jnp.int8 if 2 <= bits <= 8 and pz else None
        effective_numerics = int_numerics.IntNumerics(
            bits=bits,
            preserve_zero=pz,
            preserve_max_val=False,
            clip=True,
            round=True,
            noise_fn=None,
            clip_gradient=False,  # This can be disabled when using abs-max scaling.
            dtype=dtype,
        )

    return Tensor(
        numerics=effective_numerics,
        calib_shared_axes=None,
        scale_stop_grad=True,
        calibration=calibration.AbsMaxCalibration(),
        po2_scale=False,
        use_fake_quant=False,
        use_fwd_quant=None,
        preprocess=None,
    )


def dot_general_raw_make(
        lhs_bits=None,
        rhs_bits=None,
        local_aqt=None,
) -> 'DotGeneralRaw':
    """
    The dot_general_raw_make function is a helper function that creates a DotGeneralRaw object.

    :param lhs_bits: Determine the dtype of the lhs tensor
    :param rhs_bits: Determine the dtype of the accumulator
    :param local_aqt: Determine the type of accumulator used
    :param : Determine the dtype of the accumulator
    :return: A dotgeneralraw object
    """
    lhs_cfg = tensor_make(lhs_bits)
    rhs_cfg = tensor_make(rhs_bits)

    # Binary uses 0.5 right now.
    if (
            lhs_bits is not None
            and rhs_bits is not None
            and 2 <= lhs_bits <= 8
            and 2 <= rhs_bits <= 8
    ):
        dg_accumulator_dtype = jnp.int32
    else:
        dg_accumulator_dtype = None

    return DotGeneralRaw(
        lhs=lhs_cfg,
        rhs=rhs_cfg,
        dg_accumulator_dtype=dg_accumulator_dtype,
        local_aqt=local_aqt,
    )


def conv_general_dilated_make(
        spatial_dimensions=2,
        lhs_bits: Optional[int] = None,
        rhs_bits: Optional[int] = None,
) -> 'DotGeneralRaw':
    """Create quantization config conv_general_dilated."""
    config = dot_general_raw_make(lhs_bits, rhs_bits)
    # Hardcoding flax assumptions.
    if config.lhs:
        config.lhs.calib_shared_axes = list(range(1, spatial_dimensions + 2))
    if config.rhs:
        config.rhs.calib_shared_axes = list(range(0, spatial_dimensions + 2 - 1))
    return config


def dot_general_make(
        lhs_bits: Optional[int] = None,
        rhs_bits: Optional[int] = None,
        bwd_bits: Optional[int] = None,
        use_fwd_quant: bool = True,
        dlhs_local_aqt=None,
        drhs_local_aqt=None,
) -> 'DotGeneral':
    """Create quantization configs for input matrices to a matmul."""
    fwd = dot_general_raw_make(lhs_bits, rhs_bits)
    dlhs = dot_general_raw_make(bwd_bits, bwd_bits, local_aqt=dlhs_local_aqt)
    drhs = dot_general_raw_make(bwd_bits, bwd_bits, local_aqt=drhs_local_aqt)
    cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

    # Surprising: lhs quantization determines what drhs can do.
    if lhs_bits is not None:
        # Only rhs is accepting MultiTensor.
        cfg.drhs.rhs.use_fwd_quant = use_fwd_quant
    if rhs_bits is not None:
        cfg.dlhs.rhs.use_fwd_quant = use_fwd_quant
    return cfg


def fully_quantized(
        *,
        fwd_bits: Optional[int] = 8,
        bwd_bits: Optional[int] = 8,
        use_fwd_quant: bool = True,
        use_stochastic_rounding: Optional[bool] = True,
        # Typically we have (but it's a caller's responsibility to check):
        # - vjp_lhs_stochastic_rounding is referring to the gradient and
        # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
        vjp_lhs_stochastic_rounding: Optional[bool] = None,
        vjp_rhs_stochastic_rounding: Optional[bool] = None,
        # The dummy static bound flag is temporary, for performance benchmarking.
        use_dummy_static_bound: bool = False,
        dlhs_local_aqt: Optional[LocalQ] = None,
        drhs_local_aqt: Optional[LocalQ] = None,
) -> DotGeneral:
    """
    The fully_quantized function is a helper function that allows you to quickly
    configure the dot_general primitive with all of its quantization parameters.
    It takes in keyword arguments for each of the quantization parameters, and returns
    a DotGeneral configuration object. The following table shows what each parameter does:

    :param *: Indicate that all the parameters are keyword-only
    :param fwd_bits: Optional[int]: Specify the number of bits used for forward quantization
    :param bwd_bits: Optional[int]: Set the number of bits used for backpropagation
    :param use_fwd_quant: bool: Control whether to quantize the
    :param use_stochastic_rounding: Optional[bool]: Enable stochastic rounding
    :param vjp_lhs_stochastic_rounding: Optional[bool]: Ensure that we don't mix
    :param vjp_rhs_stochastic_rounding: Optional[bool]:
    :param use_dummy_static_bound: bool: Set the static bound to 1
    :param dlhs_local_aqt: Optional[LocalQ]: Specify the quantization scheme for the left-hand side of a matrix multiplication
    :param drhs_local_aqt: Optional[LocalQ]: Specify the quantization scheme for the right hand side of a matrix multiplication
    :param : Set the number of bits used for forward and backward pass
    :return: A dotgeneral object, which is a
    """
    cfg = dot_general_make(
        lhs_bits=fwd_bits,
        rhs_bits=fwd_bits,
        bwd_bits=bwd_bits,
        use_fwd_quant=use_fwd_quant,
        dlhs_local_aqt=dlhs_local_aqt,
        drhs_local_aqt=drhs_local_aqt,
    )

    # Stochastic Rounding
    # These 3 variables are used to ensure we don't mix
    # old and new style of SR configuration.
    old_style_sr_config = use_stochastic_rounding is not None
    new_style_sr_config_lhs = vjp_lhs_stochastic_rounding is not None
    new_style_sr_config_rhs = vjp_rhs_stochastic_rounding is not None
    assert new_style_sr_config_lhs == new_style_sr_config_rhs, (
        'if you use new style SR config (vjp_xhs_stochastic_rounding), do pass'
        ' both lhs and rhs explicitely.'
    )
    assert new_style_sr_config_lhs != old_style_sr_config

    true = True  # A crude way to get around g-explicit-bool-comparison warning

    assert not (vjp_lhs_stochastic_rounding and vjp_rhs_stochastic_rounding), (
        'This config is buggy when you set both to True. Contact lew@ or use'
        ' config_v3'
    )

    # By default use jax.uniform for stochastic rounding
    if use_stochastic_rounding == true:
        set_stochastic_rounding(cfg, True, True, 'jax.uniform')

    if vjp_lhs_stochastic_rounding == true:
        set_stochastic_rounding(cfg, True, False, 'jax.uniform')

    if vjp_rhs_stochastic_rounding == true:
        set_stochastic_rounding(cfg, False, True, 'jax.uniform')

    if use_dummy_static_bound:
        set_static_bound(cfg, 1.0)

    return cfg


def config_v3(
        *,
        fwd_bits: Optional[int] = 8,
        dlhs_bits: Optional[int] = 8,
        drhs_bits: Optional[int] = None,
        use_dummy_static_bound: bool = False,
        rng_type: str = 'jax.uniform',  # 'custom-1'
        dlhs_local_aqt: Optional[LocalQ] = None,
        drhs_local_aqt: Optional[LocalQ] = None,
        fwd_accumulator_dtype: ... = jnp.int32,
        dlhs_accumulator_dtype: ... = jnp.int32,
        drhs_accumulator_dtype: ... = None,
) -> DotGeneral:
    """
    The config_v3 function is a helper function that configures the DotGeneral
    object. It takes in keyword arguments and returns a configured DotGeneral object.
    The following are the keyword arguments:

    :param *: Indicate that all the following parameters are keyword-only
    :param fwd_bits: Optional[int]: Set the number of bits used for forward pass
    :param dlhs_bits: Optional[int]: Set the number of bits for the
    :param drhs_bits: Optional[int]: Specify the number of bits
    :param use_dummy_static_bound: bool: Set the static bound to 1
    :param rng_type: str: Specify the random number generator
    :param dlhs_local_aqt: Optional[LocalQ]: Set the local quantization of the dlhs
    :param drhs_local_aqt: Optional[LocalQ]: Set the local quantization
    :param fwd_accumulator_dtype: ...: Specify the accumulator dtype for the forward pass
    :param dlhs_accumulator_dtype: ...: Specify the accumulator dtype for the gradient
    :param drhs_accumulator_dtype: ...: Specify the data type of the accumulator in drhs
    :param : Specify the number of bits used for quantization
    :return: A dotgeneral object
    """
    fwd = dot_general_raw_make(fwd_bits, fwd_bits)
    dlhs = dot_general_raw_make(dlhs_bits, dlhs_bits, local_aqt=dlhs_local_aqt)
    drhs = dot_general_raw_make(drhs_bits, drhs_bits, local_aqt=drhs_local_aqt)
    cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

    cfg.dlhs.rhs.use_fwd_quant = False
    cfg.drhs.rhs.use_fwd_quant = False

    # Typically we have (but I don't know if it is guraranteed):
    # - vjp_lhs_stochastic_rounding is referring to the gradient and
    # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
    set_stochastic_rounding(
        cfg,
        vjp_lhs_stochastic_rounding=True,
        vjp_rhs_stochastic_rounding=False,
        implementation=rng_type,
    )

    if use_dummy_static_bound:
        set_static_bound(cfg, 1.0)

    set_accumulator_dtype(
        cfg,
        fwd_dtype=fwd_accumulator_dtype,
        dlhs_dtype=dlhs_accumulator_dtype,
        drhs_dtype=drhs_accumulator_dtype,
    )
    return cfg
