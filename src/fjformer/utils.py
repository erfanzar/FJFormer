"""Utility functions for JAX."""
import logging
import importlib.util
from typing import Union, Tuple

import flax
import jax
from jax import random as jrandom


def is_torch_available() -> bool:
    """Checks if PyTorch is installed.

    Returns:
        True if the torch module is installed, False otherwise.
    """
    return importlib.util.find_spec("torch") is not None


def count_num_params(params: flax.core.frozen_dict.FrozenDict) -> int:
    """Counts the number of parameters in a model.

    Args:
        params: A Flax FrozenDict containing the model parameters.

    Returns:
        The total number of parameters in the model.
    """
    return sum(
        i.size for i in jax.tree_util.tree_flatten(flax.core.unfreeze(params))[0]
    )


def count_params(params: flax.core.frozen_dict.FrozenDict) -> None:
    """Prints the number of parameters in a Flax model.

    Args:
        params: A Flax FrozenDict containing the model parameters.
    """
    print(
        f"\033[1;31mModel Contains : {count_num_params(params) / 1e9:.2f} Billion Parameters"
    )


class JaxRNG:
    """A wrapper around JAX's PRNGKey that simplifies key splitting."""

    def __init__(self, rng: jrandom.PRNGKey):
        """Initializes the JaxRNG with a PRNGKey.

        Args:
            rng: A JAX PRNGKey.
        """
        self.rng = rng

    @classmethod
    def from_seed(cls, seed: int) -> "JaxRNG":
        """Creates a JaxRNG instance from a seed.

        Args:
            seed: The seed to use for the random number generator.

        Returns:
            A JaxRNG instance.
        """
        return cls(jrandom.PRNGKey(seed))

    def __call__(
        self, keys: Union[int, Tuple[str, ...]] = None
    ) -> Union[jrandom.PRNGKey, Tuple[jrandom.PRNGKey, ...], dict]:
        """Splits the internal PRNGKey and returns new keys.

        Args:
            keys:  If None, returns a single split key and updates the internal RNG.
                   If an int, splits the key into `keys + 1` parts, updates the internal RNG,
                   and returns the last `keys` parts as a tuple.
                   If a tuple of strings, splits the key into `len(keys) + 1` parts,
                   updates the internal RNG, and returns a dictionary mapping the strings
                   to their corresponding key parts.

        Returns:
            The split PRNGKey(s) based on the `keys` argument.
        """
        if keys is None:
            self.rng, split_rng = jrandom.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jrandom.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jrandom.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


# Global JaxRNG instance
jax_utils_rng = None


def init_rng(seed: int) -> None:
    """Initializes the global JaxRNG with a seed.

    Args:
        seed: The seed to use for the random number generator.
    """
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(
    *args, **kwargs
) -> Union[jrandom.PRNGKey, Tuple[jrandom.PRNGKey, ...], dict]:
    """Provides access to the global JaxRNG and splits the key based on arguments.

    This function wraps the global `jax_utils_rng` instance and calls its `__call__` method,
    passing through any arguments provided. This provides a convenient way to access and
    split the global random number generator key.

    Args:
        *args: Positional arguments passed to the `jax_utils_rng` instance's `__call__` method.
        **kwargs: Keyword arguments passed to the `jax_utils_rng` instance's `__call__` method.

    Returns:
        The split PRNGKey(s) from the global `jax_utils_rng` instance.
    """
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


class GenerateRNG:
    """An infinite generator of JAX PRNGKeys, useful for iterating over seeds."""

    def __init__(self, seed: int = 0):
        """Initializes the generator with a starting seed.

        Args:
            seed: The seed to use for the initial PRNGKey.
        """
        self.seed = seed
        self._rng = jrandom.PRNGKey(seed)

    def __next__(self) -> jrandom.PRNGKey:
        """Generates and returns the next PRNGKey in the sequence.

        Returns:
            The next PRNGKey derived from the internal state.
        """
        self._rng, key = jrandom.split(self._rng)
        return key

    @property
    def rng(self) -> jrandom.PRNGKey:
        """Provides access to the next PRNGKey without advancing the generator.

        Returns:
            The next PRNGKey in the sequence.
        """
        return next(self)


def get_logger(name, level: int = logging.INFO) -> logging.Logger:
    """
    Function to create and configure a logger.
    Args:
        name: str: The name of the logger.
        level: int: The logging level. Defaults to logging.INFO.
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    # Set the logging level
    logger.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def set_loggers_level(level: int = logging.WARNING):
    """Function to set the logging level of all loggers to the specified level.

    Args:
        level: int: The logging level to set. Defaults to
            logging.WARNING.
    """
    logging.root.setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)
