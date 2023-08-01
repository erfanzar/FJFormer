import jax.numpy as np
from jax.scipy.special import logsumexp
import jax


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
