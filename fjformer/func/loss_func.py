import chex
import jax.numpy as np
from jax.scipy.special import logsumexp
import jax
from jax import numpy as jnp


# Mean Squared Error
def mse(labels, predictions):
    return np.mean((labels - predictions) ** 2)


# Mean Absolute Error
def mae(labels, predictions):
    return np.mean(np.abs(labels - predictions))


# Cross Entropy
def cross_entropy(labels, predictions, ignore_index=None):
    labels = jax.nn.one_hot(labels, predictions.shape[-1])
    if ignore_index is not None:
        mask = np.ones_like(labels)
        mask = np.where(labels == ignore_index, 0, mask)
        labels = labels * mask
        predictions = predictions * mask
    log_softmax = predictions - logsumexp(predictions, axis=-1, keepdims=True)
    return -np.sum(labels * log_softmax) / labels.shape[0]


# Binary Cross Entropy
def binary_cross_entropy(labels, predictions):
    labels = jax.nn.one_hot(labels, predictions.shape[-1])
    return -np.mean(labels * np.log(predictions + 1e-8) + (1 - labels) * np.log(1 - predictions + 1e-8))


# Negative Log Likelihood
def nll(labels, predictions):
    return -np.sum(labels * np.log(predictions + 1e-8))


# L2 Loss
def l2(labels, predictions):
    return np.sum((labels - predictions) ** 2)


# Hinge Loss
def hinge(labels, predictions):
    return np.mean(np.maximum(0, 1 - labels * predictions))


# Log-Cosh Loss
def log_cosh(labels, predictions):
    def cosh(x):
        return (np.exp(x) + np.exp(-x)) / 2

    return np.mean(np.log(cosh(predictions - labels)))


def binary_cross_entropy_onehot(labels, predictions):
    labels = jax.nn.one_hot(labels, predictions.shape[-1])
    return -np.mean(labels * np.log(predictions + 1e-8) + (1 - labels) * np.log(1 - predictions + 1e-8))


def cross_entropy_onehot(labels, predictions):
    labels = jax.nn.one_hot(labels, predictions.shape[-1])
    log_softmax = predictions - logsumexp(predictions, axis=-1, keepdims=True)
    return -np.sum(labels * log_softmax) / labels.shape[0]


def mse_loss(val, target, valid=None):
    if valid is None:
        valid = jnp.ones((*target.shape[:2], 1))
    valid = valid.astype(jnp.float32)
    loss = jnp.mean(
        jnp.where(
            valid > 0.0,
            jnp.square(val - target),
            0.0
        )
    )
    return loss


def cross_entropy_loss_and_accuracy(logits: chex.Array, tokens: chex.Array, valid: chex.Array = None):
    """

    :param logits: Logits
    :param tokens: labels
    :param valid: attention_mask
    :return: loss and accuracy
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
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy


def fused_cross_entropy_loss_and_accuracy(logits: chex.Array, tokens: chex.Array, valid: chex.Array = None):
    """

    :param logits: Logits
    :param tokens: labels
    :param valid: attention_mask
    :return: loss and accuracy
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
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy
