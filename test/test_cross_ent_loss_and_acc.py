import jax.numpy as jnp
from fjformer.func.loss_func import (
    compute_weighted_cross_entropy_and_accuracy,
    cross_entropy_loss_and_accuracy,
    cross_entropy_with_logits,
    get_loss_normalizing_factor_and_weights,
    fused_cross_entropy_loss_and_accuracy,
    SpecialLossNormalizingFactor,
)
import numpy as np
from optax import softmax_cross_entropy


def test_compute_weighted_cross_entropy_1():
    """
    batch_size = 2
    length = 4
    vocab_size = 8

    :return:
    """
    label_smoothing = 0.0
    z_loss_coeff = 0.0001

    logits = jnp.array(
        [
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                [0.2, 0.3, 0.5, 0.7, 0.11, 0.13, 0.17, 0.19],
                [0.19, 0.17, 0.13, 0.11, 0.7, 0.5, 0.3, 0.2],
            ],
            [
                [0.5, 0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6],
                [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.8, 0.7],
                [0.3, 0.5, 0.7, 0.11, 0.13, 0.17, 0.19, 0.2],
                [0.2, 0.19, 0.17, 0.13, 0.11, 0.7, 0.5, 0.3],
            ],
        ],
        dtype=jnp.float32,
    )
    targets = jnp.array(
        [
            [7, 1, 3, 3],  # have two correct predictions based on highest logits
            [5, 6, 2, 7],  # have two correct predictions
        ],
        dtype=jnp.int32,
    )
    decoder_loss_weights = jnp.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=jnp.float32)

    batch = {
        "decoder_target_tokens": targets,
        "decoder_loss_weights": decoder_loss_weights,
    }
    loss_normalizing_factor = SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
    lnf, weights = get_loss_normalizing_factor_and_weights(
        loss_normalizing_factor, batch
    )

    (
        wce_loss,
        wce_z_loss,
        wce_weight_sum,
        wce_accuracy,
    ) = compute_weighted_cross_entropy_and_accuracy(
        logits=logits,
        targets=targets,
        weights=weights,
        label_smoothing=label_smoothing,
        z_loss=z_loss_coeff,
        loss_normalizing_factor=lnf,
    )

    print(
        f"Loss: {wce_loss}, Z-Loss: {wce_z_loss}, Weight Sum: {wce_weight_sum}, Accuracy: {wce_accuracy}"
    )
    ce_loss, ce_accuracy = cross_entropy_loss_and_accuracy(logits, targets)
    print(f"Cross Entropy Loss: {ce_loss}, Accuracy: {wce_accuracy}")

    assert np.isclose(ce_loss, wce_loss, rtol=1e-3, atol=1e-3), "Cross Entropy Losses do not match"
    assert np.isclose(ce_accuracy, wce_accuracy, rtol=1e-2, atol=1e-2), "Accuracies do not match"


def test_compute_weighted_cross_entropy_ignore_token_minus100():
    """
    batch_size = 2
    length = 4
    vocab_size = 8

    :return:
    """
    label_smoothing = 0.0
    z_loss_coeff = 0.0000

    logits = jnp.array(
        [
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                [0.2, 0.3, 0.5, 0.7, 0.11, 0.13, 0.17, 0.19],
                [0.19, 0.17, 0.13, 0.11, 0.7, 0.5, 0.3, 0.2],
            ],
            [
                [0.5, 0.4, 0.3, 0.2, 0.1, 0.8, 0.7, 0.6],
                [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.8, 0.7],
                [0.3, 0.5, 0.7, 0.11, 0.13, 0.17, 0.19, 0.2],
                [0.2, 0.19, 0.17, 0.13, 0.11, 0.7, 0.5, 0.3],
            ],
        ],
        dtype=jnp.float32,
    )
    targets = jnp.array(
        [
            [-100, 1, 3, 3],  # Ignore first token
            [-100, -100, 2, 7],  # Ignore first two tokens
        ],
        dtype=jnp.int32,
    )
    decoder_loss_weights = (
        None  # Will be calculated by get_loss_normalizing_factor_and_weights
    )

    batch = {
        "decoder_target_tokens": targets,
        "decoder_loss_weights": decoder_loss_weights,
    }
    loss_normalizing_factor = SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
    lnf, weights = get_loss_normalizing_factor_and_weights(
        loss_normalizing_factor, batch
    )

    (
        wce_loss,
        wce_z_loss,
        wce_weight_sum,
        wce_accuracy,
    ) = compute_weighted_cross_entropy_and_accuracy(
        logits=logits,
        targets=targets,
        weights=weights,
        label_smoothing=label_smoothing,
        z_loss=z_loss_coeff,
        loss_normalizing_factor=lnf,
    )

    print(
        f"Loss: {wce_loss}, Z-Loss: {wce_z_loss}, Weight Sum: {wce_weight_sum}, Accuracy: {wce_accuracy}"
    )

    # Construct valid array based on targets
    valid = jnp.where(targets > 0, 1, 0)
    ce_loss, ce_accuracy = cross_entropy_loss_and_accuracy(logits, targets, valid)
    print(f"Cross Entropy Loss: {ce_loss}, Accuracy: {ce_accuracy}")

    assert np.isclose(ce_loss, wce_loss, rtol=1e-2, atol=1e-2), "Cross Entropy Losses do not match"
    assert np.isclose(ce_accuracy, wce_accuracy, rtol=1e-1, atol=1e-1), "Accuracies do not match"


def test_multiple():
    # Logits: Unnormalized log probabilities from a model, shaped as [batch_size, num_classes].
    # Here, batch_size=2 for two examples, and num_classes=3 for three possible classes.
    logits = jnp.array(
        [
            [2.0, 1.0, 0.1],  # Logits for the first example
            [0.1, 2.0, 1.0],  # Logits for the second example
        ]
    )

    # Labels: One-hot encoded true class labels, with the same shape as logits [batch_size, num_classes].
    # Each row is a one-hot vector representing the true class of each example.
    labels = jnp.array(
        [
            [1, 0, 0],  # True class is the first one for the first example
            [0, 1, 0],  # True class is the second one for the second example
        ]
    )

    # Convert one-hot encoded labels to class indices (tokens)
    # Tokens are class indices representing the true classes for each example in the batch.
    # Shaped as [batch_size], where each entry is the index (0-based) of the true class.
    tokens = jnp.argmax(labels, axis=1)

    # Create a valid array  (a.k.a. decoder_loss_weights) that matches the shape of logits
    # Here, we're assuming all positions in all sequences are valid
    valid = jnp.ones_like(
        tokens, dtype=jnp.float32
    )  # Shape now correctly matches logits: (2, 3)
    print(f"Tokens: {tokens}, Valid: {valid}, valid_shape: {valid.shape}")

    # Computation using logits and class indices (tokens) as the representation of true labels.
    ce_loss, ce_accuracy = cross_entropy_loss_and_accuracy(logits, tokens, valid)
    print(f"Cross Entropy Loss (Function 1): {ce_loss}, Accuracy: {ce_accuracy}")

    # Computation using logits and class indices (tokens) as the representation of true labels.
    fused_ce_loss, fused_accuracy = fused_cross_entropy_loss_and_accuracy(
        logits, tokens, valid
    )
    print(
        f"Fused Cross Entropy Loss (Function 2): {fused_ce_loss}, Accuracy: {fused_accuracy}"
    )

    # Computation using logits and one-hot encoded labels to calculate the loss for each example.
    softmax_ce_loss = softmax_cross_entropy(logits, labels)
    print(f"Softmax Cross Entropy Loss (Function 2): {softmax_ce_loss.mean()}")

    z_loss = 0.0001
    total_loss, total_z_loss = cross_entropy_with_logits(logits, labels, z_loss)
    print(total_loss, total_z_loss)
    print(
        f"Cross Entropy with Logits Loss: {total_loss.mean()}, Z-loss: {total_z_loss.mean()}"
    )

    decoder_loss_weights = jnp.ones_like(tokens, dtype=jnp.float32)

    batch = {
        "decoder_target_tokens": tokens,
        "decoder_loss_weights": decoder_loss_weights,
    }

    label_smoothing = 0.0
    z_loss = 0.0001

    loss_normalizing_factor = SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
    lnf, weights = get_loss_normalizing_factor_and_weights(
        loss_normalizing_factor, batch
    )

    (
        total_loss,
        total_z_loss,
        weight_sum,
        ce_accuracy,
    ) = compute_weighted_cross_entropy_and_accuracy(
        logits=logits,
        targets=batch["decoder_target_tokens"],
        weights=weights,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=lnf,
    )

    print(
        f"Weighted Cross Entropy Loss: {total_loss}, Z-loss: {total_z_loss}, Weight Sum: {weight_sum}, Accuracy: {ce_accuracy}"
    )

    assert np.isclose(ce_loss, total_loss, rtol=1e-2, atol=1e-2), "Cross Entropy Losses do not match"
    assert np.isclose(ce_accuracy, fused_accuracy, rtol=1e-2, atol=1e-2), "Accuracies do not match"


if __name__ == "__main__":
    test_compute_weighted_cross_entropy_ignore_token_minus100()
    test_compute_weighted_cross_entropy_1()
    test_multiple()
