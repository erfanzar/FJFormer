import jax.numpy as np
from jax.scipy.special import logsumexp
import jax


# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Mean Absolute Error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# Cross Entropy
def cross_entropy(y_true, y_pred, ignore_index=None):
    y_true = jax.nn.one_hot(y_true, y_pred.shape[-1])
    if ignore_index is not None:
        mask = np.ones_like(y_true)
        mask = np.where(y_true == ignore_index, 0, mask)
        y_true = y_true * mask
        y_pred = y_pred * mask
    log_softmax = y_pred - logsumexp(y_pred, axis=-1, keepdims=True)
    return -np.sum(y_true * log_softmax) / y_true.shape[0]


# Binary Cross Entropy
def binary_cross_entropy(y_true, y_pred):
    y_true = jax.nn.one_hot(y_true, y_pred.shape[-1])
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))


# Negative Log Likelihood
def nll(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8))


# L2 Loss
def l2(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


# Hinge Loss
def hinge(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))


# Log-Cosh Loss
def log_cosh(y_true, y_pred):
    def cosh(x):
        return (np.exp(x) + np.exp(-x)) / 2

    return np.mean(np.log(cosh(y_pred - y_true)))
