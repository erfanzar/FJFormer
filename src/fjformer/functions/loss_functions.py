import enum
from functools import reduce
from operator import mul
from typing import Mapping, Optional, Tuple, Union

import chex
import jax
import jax.numpy as np
from flax.training import common_utils
from jax import Array
from jax import numpy as jnp
from jax.scipy.special import logsumexp

# Mean Squared Error


def mse(labels: Array, predictions: Array) -> float:
    """
    Computes the mean squared error between two arrays.

    Args:
        labels: The true values.
        predictions: The predicted values.

    Returns:
        The mean squared error (MSE) between the labels and predictions.
    """
    return np.mean((labels - predictions) ** 2)


# Mean Absolute Error
def mae(labels: Array, predictions: Array) -> float:
    """
    Computes the mean absolute error between two arrays.

    Args:
        labels: The true values.
        predictions: The predicted values.

    Returns:
        The mean absolute error (MAE) between the labels and predictions.
    """
    return np.mean(np.abs(labels - predictions))


# Cross Entropy
def cross_entropy(
    labels: Array, predictions: Array, ignore_index: Optional[int] = None
) -> float:
    """
    Computes the cross-entropy loss between labels and predictions.

    Args:
        labels: The true class labels (integers).
        predictions: The predicted class probabilities.
        ignore_index: An optional index to ignore when computing the loss.

    Returns:
        The cross-entropy loss.
    """
    labels = jax.nn.one_hot(labels, predictions.shape[-1])
    if ignore_index is not None:
        mask = np.ones_like(labels)
        mask = np.where(labels == ignore_index, 0, mask)
        labels = labels * mask
        predictions = predictions * mask
    log_softmax = predictions - logsumexp(predictions, axis=-1, keepdims=True)
    return -np.sum(labels * log_softmax) / labels.shape[0]


# Binary Cross Entropy
def binary_cross_entropy(labels: Array, predictions: Array) -> float:
    """
    Computes the binary cross-entropy loss between labels and predictions.

    Args:
        labels: The true binary labels (0 or 1).
        predictions: The predicted probabilities for the positive class.

    Returns:
        The binary cross-entropy loss.
    """
    labels = jax.nn.one_hot(labels, predictions.shape[-1])
    return -np.mean(
        labels * np.log(predictions + 1e-8)
        + (1 - labels) * np.log(1 - predictions + 1e-8)
    )


# Negative Log Likelihood
def nll(labels: Array, predictions: Array) -> float:
    """
    Computes the negative log-likelihood loss.

    Args:
        labels: The true class labels (integers).
        predictions: The predicted class probabilities.

    Returns:
        The negative log-likelihood loss.
    """
    return -np.sum(labels * np.log(predictions + 1e-8))


# L2 Loss
def l2(labels: Array, predictions: Array) -> float:
    """
    Computes the L2 loss (sum of squared differences).

    Args:
        labels: The true values.
        predictions: The predicted values.

    Returns:
        The L2 loss.
    """
    return np.sum((labels - predictions) ** 2)


# Hinge Loss
def hinge(labels: Array, predictions: Array) -> float:
    """
    Computes the hinge loss, commonly used in SVMs.

    Args:
        labels: The true class labels (-1 or 1).
        predictions: The predicted values.

    Returns:
        The hinge loss.
    """
    return np.mean(np.maximum(0, 1 - labels * predictions))


# Log-Cosh Loss
def log_cosh(labels: Array, predictions: Array) -> float:
    """
    Computes the log-cosh loss, a smooth approximation of the MAE.

    Args:
        labels: The true values.
        predictions: The predicted values.

    Returns:
        The log-cosh loss.
    """

    def cosh(x):
        return (np.exp(x) + np.exp(-x)) / 2

    return np.mean(np.log(cosh(predictions - labels)))


def binary_cross_entropy_onehot(labels: Array, predictions: Array) -> float:
    """
    Computes the binary cross-entropy loss using one-hot encoded labels.

    Args:
        labels: The true class labels (one-hot encoded).
        predictions: The predicted class probabilities.

    Returns:
        The binary cross-entropy loss.
    """
    labels = jax.nn.one_hot(labels, predictions.shape[-1])
    return -np.mean(
        labels * np.log(predictions + 1e-8)
        + (1 - labels) * np.log(1 - predictions + 1e-8)
    )


def cross_entropy_onehot(labels: Array, predictions: Array) -> float:
    """
    Computes the cross-entropy loss using one-hot encoded labels.

    Args:
        labels: The true class labels (integers).
        predictions: The predicted class probabilities.

    Returns:
        The cross-entropy loss.
    """
    labels = jax.nn.one_hot(labels, predictions.shape[-1])
    log_softmax = predictions - logsumexp(predictions, axis=-1, keepdims=True)
    return -np.sum(labels * log_softmax) / labels.shape[0]


def mse_loss(val: Array, target: Array, valid: Optional[Array] = None) -> float:
    """
    Computes the mean squared error loss with optional masking.

    Args:
        val: The predicted values.
        target: The true values.
        valid: An optional mask to indicate valid positions.

    Returns:
        The masked mean squared error loss.
    """
    if valid is None:
        valid = jnp.ones((*target.shape[:2], 1))
    valid = valid.astype(jnp.float32)
    loss = jnp.mean(jnp.where(valid > 0.0, jnp.square(val - target), 0.0))
    return loss


def cross_entropy_loss_and_accuracy(
    logits: chex.Array, tokens: chex.Array, valid: Optional[chex.Array] = None
) -> Tuple[float, float]:
    """
    Computes the cross-entropy loss and accuracy with optional masking.

    Args:
        logits: The predicted logits.
        tokens: The true class labels (integers).
        valid: An optional mask to indicate valid positions.

    Returns:
        A tuple containing the loss and accuracy.
    """
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32)  # for numerical stability
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        valid > 0.0, jnp.argmax(logits, axis=-1) == tokens, jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy


def fused_cross_entropy_loss_and_accuracy(
    logits: chex.Array, tokens: chex.Array, valid: Optional[chex.Array] = None
) -> Tuple[float, float]:
    """
    Computes the cross-entropy loss and accuracy with optional masking (fused version).

    Args:
        logits: The predicted logits.
        tokens: The true class labels (integers).
        valid: An optional mask to indicate valid positions.

    Returns:
        A tuple containing the loss and accuracy.
    """
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            log_probs,
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        valid > 0.0, jnp.argmax(logits, axis=-1) == tokens, jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy


@jax.custom_vjp
def cross_entropy_with_logits(
    logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes cross-entropy loss with a stable custom gradient.

    This function computes the cross-entropy loss with an optional auxiliary loss
    term (`z_loss`) to prevent logits from drifting too far from zero.

    Args:
        logits: The predicted logits.
        targets: The one-hot encoded target labels.
        z_loss: Coefficient for the auxiliary z-loss term.

    Returns:
        A tuple containing the total loss and the z_loss.
    """
    logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    log_softmax = logits - logits_sum
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    # Add auxiliary z-loss term.
    log_z = jnp.squeeze(logits_sum, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    loss += total_z_loss
    return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
    logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0.0
) -> Tuple[Tuple[Array, Array], Tuple[Array, Array, float, Array, Array, Array, Array]]:
    """Forward-mode of `cross_entropy_with_logits`."""
    max_logit = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_softmax = shifted - jnp.log(sum_exp)
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    # Add auxiliary z-loss term.
    log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    loss += total_z_loss
    return (loss, total_z_loss), (
        logits,
        targets,
        z_loss,
        exp_shifted,
        sum_exp,
        log_softmax,
        log_z,
    )


def _cross_entropy_with_logits_bwd(
    res: Tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    g: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward-mode of `cross_entropy_with_logits`."""
    g = g[0]  # Ignore z_loss component as that is only used for logging.
    logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
    # z-loss term adds the (2 * z_loss * log_z) factor.
    deriv = (
        jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
    )
    g_logits = jnp.expand_dims(g, axis=-1) * deriv
    g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
    return (
        jnp.asarray(g_logits, logits.dtype),
        jnp.asarray(g_targets, targets.dtype),
        jnp.array(0.0),
    )  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(
    _cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd
)


def compute_weighted_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    loss_normalizing_factor: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes weighted cross-entropy loss, z-loss, and weight sum.

    Args:
        logits: The predicted logits.
        targets: The target class labels (integers).
        weights: Optional weights for each example.
        label_smoothing: Label smoothing factor.
        z_loss: Coefficient for the auxiliary z-loss term.
        loss_normalizing_factor: A factor to normalize the loss.

    Returns:
        A tuple containing the total loss, z-loss, and sum of weights.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence)
        + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    soft_targets = common_utils.onehot(
        targets, vocab_size, on_value=confidence, off_value=low_confidence
    )
    total_loss, total_z_loss = cross_entropy_with_logits(
        logits, soft_targets, z_loss=z_loss
    )
    total_loss = total_loss - normalizing_constant

    shape_dtype_struct = jax.eval_shape(lambda x: x, targets)
    weight_sum = reduce(mul, shape_dtype_struct.shape, 1)
    if weights is not None:
        total_loss = total_loss * weights
        total_z_loss = total_z_loss * weights
        weight_sum = jnp.sum(weights)

    # By default, we do not normalize loss based on anything.
    # We don't normalize based on batch size because the optimizers we use are
    # pretty much scale invariant, so this simplifies things.
    # We don't normalize based on the number of non-padding tokens in order to treat
    # each token as equally important regardless of sequence length.
    if loss_normalizing_factor is not None:
        total_loss /= loss_normalizing_factor
        total_z_loss /= loss_normalizing_factor
    return jnp.sum(total_loss), jnp.sum(total_z_loss), weight_sum


def compute_weighted_cross_entropy_and_accuracy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    loss_normalizing_factor: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes weighted cross-entropy loss, z-loss, weight sum, and accuracy.

    Args:
        logits: The predicted logits.
        targets: The target class labels (integers).
        weights: Optional weights for each example.
        label_smoothing: Label smoothing factor.
        z_loss: Coefficient for the auxiliary z-loss term.
        loss_normalizing_factor: A factor to normalize the loss.

    Returns:
        A tuple containing the total loss, z-loss, sum of weights, and accuracy.
    """
    total_loss, total_z_loss, weight_sum = compute_weighted_cross_entropy(
        logits, targets, weights, label_smoothing, z_loss, loss_normalizing_factor
    )

    predictions = jnp.argmax(logits, axis=-1)
    correct_predictions = jnp.equal(predictions, targets).astype(jnp.float32)
    accuracy = jnp.sum(correct_predictions * weights) / weight_sum

    return total_loss, total_z_loss, weight_sum, accuracy


@enum.unique
class SpecialLossNormalizingFactor(enum.Enum):
    """
    Specially calculated loss normalizing factors that are not constant.

    Attributes:
        NUM_REAL_TARGET_TOKENS: Divide the loss by the number of real (non-padding) tokens.
        NUM_TOTAL_TARGET_TOKENS: Divide the loss by the total number of target tokens.
        AVERAGE_PER_SEQUENCE: Compute the average loss per sequence.
    """

    NUM_REAL_TARGET_TOKENS = 1
    NUM_TOTAL_TARGET_TOKENS = 2
    AVERAGE_PER_SEQUENCE = 3


def convert_special_loss_normalizing_factor_to_enum(
    x: str,
) -> SpecialLossNormalizingFactor:
    """
    Converts a stringified version of SpecialLossNormalizingFactor to an enum.

    Args:
        x: Stringified version of the enum value.

    Returns:
        The corresponding SpecialLossNormalizingFactor enum value.
    """
    x = x.upper()
    if x == "NUM_REAL_TARGET_TOKENS":
        return SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
    if x == "NUM_TOTAL_TARGET_TOKENS":
        return SpecialLossNormalizingFactor.NUM_TOTAL_TARGET_TOKENS
    if x == "AVERAGE_PER_SEQUENCE":
        return SpecialLossNormalizingFactor.AVERAGE_PER_SEQUENCE
    raise ValueError(
        'Could not convert string "%s" to SpecialLossNormalizingFactor' % x
    )


@jax.vmap
def _sum_weights_per_segment(
    positions: jnp.ndarray, segment_ids: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    """Sums weights per packed segment to produce a normalizing vector."""

    # NB: Assumes padding only occurs at the end of a sequence.

    def _repeat_last_nonnegative(xs, reverse=False):
        def fn(prev, x):
            y = jnp.where(x == 0, prev, x)
            return y, y

        return jax.lax.scan(fn, jnp.zeros_like(xs[0]), xs, reverse=reverse)[1]

    # Compute final positions per sequence.
    start_positions = positions == 0
    final_positions = jnp.concatenate([start_positions[1:], jnp.ones(1)])
    # Clear padded positions.
    final_positions *= segment_ids != 0
    # Compute cumulative weights, clearing all but final position per sequence.
    final_cumulative_weights = final_positions * jnp.cumsum(weights)
    # Subtract sequences' final weights from cumulative weights of following ones.
    final_total_weights = jnp.concatenate(
        [
            final_cumulative_weights[0:1],
            jnp.diff(_repeat_last_nonnegative(final_cumulative_weights)),
        ]
    )
    # Copy final sequence weight to all positions in sequence.
    normalizer = _repeat_last_nonnegative(final_total_weights, reverse=True)
    return normalizer


def get_loss_normalizing_factor_and_weights(
    loss_normalizing_factor: Optional[
        Union[float, int, str, SpecialLossNormalizingFactor]
    ],
    batch: Mapping[str, jnp.ndarray],
) -> Tuple[Optional[float], Optional[jnp.ndarray]]:
    """
    Gets the loss normalizing factor and weights from a batch of data.

    Args:
        loss_normalizing_factor: The loss normalizing factor to use.
        batch: A dictionary containing the input batch of data.

    Returns:
        A tuple containing the loss normalizing factor and loss weights.
    """
    loss_weights = batch.get("decoder_loss_weights", None)
    if loss_normalizing_factor is None or not isinstance(
        loss_normalizing_factor, (str, SpecialLossNormalizingFactor)
    ):
        return loss_normalizing_factor, loss_weights

    if isinstance(loss_normalizing_factor, str):
        loss_normalizing_factor = convert_special_loss_normalizing_factor_to_enum(
            loss_normalizing_factor
        )

    # If `loss_weights` are not provided, we assume that the padding id is 0 and
    # that non-padding tokens in the decoder all correspond to the positions
    # where loss should be taken. If more fine-grained behavior (e.g., taking
    # loss on subset of 'decoder_target_tokens') is desired, provide
    # `loss_weights` that account for this.
    if loss_weights is None:
        loss_weights = jnp.asarray(batch["decoder_target_tokens"] > 0, jnp.float32)

    output_normalizing_factor = None
    if loss_normalizing_factor == SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS:
        output_normalizing_factor = jnp.sum(loss_weights)
    elif (
        loss_normalizing_factor == SpecialLossNormalizingFactor.NUM_TOTAL_TARGET_TOKENS
    ):
        output_normalizing_factor = np.prod(batch["decoder_target_tokens"].shape)
    elif loss_normalizing_factor == SpecialLossNormalizingFactor.AVERAGE_PER_SEQUENCE:
        if "decoder_segment_ids" in batch:  # is packed
            norm_vec = _sum_weights_per_segment(
                batch["decoder_positions"], batch["decoder_segment_ids"], loss_weights
            )
        else:
            norm_vec = jnp.sum(loss_weights, axis=-1, keepdims=True)
        # Handle divide-by-zero.
        loss_weights = jnp.nan_to_num(
            loss_weights / norm_vec, nan=0, posinf=0, neginf=0
        )
        output_normalizing_factor = jnp.sum(loss_weights)
    else:
        raise ValueError(
            "Unsupported value of loss_normalizing_factor: %s"
            % str(loss_normalizing_factor)
        )

    return output_normalizing_factor, loss_weights


def auxiliary_load_balancing_loss_func(
    gate_logits: chex.Array,
    num_experts: int,
    top_k: int,
    attention_mask: Optional[chex.Array] = None,
) -> chex.Array:
    """Computes auxiliary load balancing loss as in Switch Transformer.

    See Switch Transformer (https://arxiv.org/abs/2101.03961)

    Args:
        gate_logits: The logits for the gating network.
        num_experts: The number of experts.
        top_k: The number of experts to select.
        attention_mask: An optional attention mask.

    Returns:
        The auxiliary load balancing loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return jnp.array(0.0, dtype=jnp.float32)
    elif isinstance(gate_logits, tuple):
        concatenated_gate_logits = jnp.concatenate(gate_logits, axis=0)
    else:
        return jnp.array(0.0, dtype=jnp.float32)
    routing_weights = jax.nn.softmax(concatenated_gate_logits, axis=-1)

    _, selected_experts = jax.lax.top_k(routing_weights, top_k)

    expert_mask = jax.nn.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        tokens_per_expert = jnp.mean(expert_mask.astype(jnp.float32), axis=0)
        router_prob_per_expert = jnp.mean(routing_weights, axis=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        expert_attention_mask = jnp.broadcast_to(
            attention_mask[jnp.newaxis, :, :, jnp.newaxis, jnp.newaxis],
            (num_hidden_layers, batch_size, sequence_length, top_k, num_experts),
        ).reshape(-1, top_k, num_experts)

        tokens_per_expert = jnp.sum(
            expert_mask.astype(jnp.float32) * expert_attention_mask, axis=0
        ) / jnp.sum(expert_attention_mask, axis=0)

        router_per_expert_attention_mask = jnp.broadcast_to(
            attention_mask[jnp.newaxis, :, :, jnp.newaxis],
            (num_hidden_layers, batch_size, sequence_length, num_experts),
        ).reshape(-1, num_experts)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = jnp.sum(
            routing_weights * router_per_expert_attention_mask, axis=0
        ) / jnp.sum(router_per_expert_attention_mask, axis=0)

    overall_loss = jnp.sum(
        tokens_per_expert * jnp.expand_dims(router_prob_per_expert, 0)
    )
    return overall_loss * num_experts
