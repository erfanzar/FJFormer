from fjformer.functions._func import average_metrics, global_norm, transpose, fused_softmax
from fjformer.functions.loss_func import (
    auxiliary_load_balancing_loss_func as auxiliary_load_balancing_loss_func,
    get_loss_normalizing_factor_and_weights as get_loss_normalizing_factor_and_weights,
    convert_special_loss_normalizing_factor_to_enum as convert_special_loss_normalizing_factor_to_enum,
    SpecialLossNormalizingFactor as SpecialLossNormalizingFactor,
    cross_entropy_loss_and_accuracy as cross_entropy_loss_and_accuracy,
    fused_cross_entropy_loss_and_accuracy as fused_cross_entropy_loss_and_accuracy,
    compute_weighted_cross_entropy_and_accuracy as compute_weighted_cross_entropy_and_accuracy,
    compute_weighted_cross_entropy as compute_weighted_cross_entropy,
    binary_cross_entropy_onehot as binary_cross_entropy_onehot,
    binary_cross_entropy as binary_cross_entropy,
    cross_entropy as cross_entropy,
    cross_entropy_with_logits as cross_entropy_with_logits,
    cross_entropy_onehot as cross_entropy_onehot,
    l2 as l2,
    hinge as hinge,
    mae as mae,
    nll as nll,
    mse_loss as mse_loss,
    mse as mse,
)

__all__ = (
    "average_metrics",
    "global_norm",
    "transpose",
    "fused_softmax",
    "auxiliary_load_balancing_loss_func",
    "get_loss_normalizing_factor_and_weights",
    "convert_special_loss_normalizing_factor_to_enum",
    "SpecialLossNormalizingFactor",
    "cross_entropy_loss_and_accuracy",
    "fused_cross_entropy_loss_and_accuracy",
    "compute_weighted_cross_entropy_and_accuracy",
    "compute_weighted_cross_entropy",
    "binary_cross_entropy_onehot",
    "binary_cross_entropy",
    "cross_entropy",
    "cross_entropy_with_logits",
    "cross_entropy_onehot",
    "l2",
    "hinge",
    "mae",
    "mse_loss",
    "mse",
    "nll",
)
