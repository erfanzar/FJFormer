import importlib.util
import jax
import flax


def is_torch_available():
    """
    The is_torch_available function checks if PyTorch is installed.

    :return: True if the torch module is installed
    
    """
    return True if importlib.util.find_spec('torch') is not None else False


def count_num_params(_p):
    """
    The count_num_params function is a helper function that counts the number of parameters in a model.
    It takes as input an unfrozen parameter dictionary, and returns the total number of parameters.


    :param _p: Count the number of parameters in a model
    :return: The number of parameters in the model
    
    """
    return sum(i.size for i in jax.tree_util.tree_flatten(flax.core.unfreeze(_p))[0])


def count_params(_p):
    """
    The count_params function takes in a Flax model and prints out the number of parameters it contains.
        Args:
            _p (Flax Params]): A Flax model to count the number of parameters for.

    :param _p: Count the number of parameters in a model
    :return: The number of parameters in a model
    
    """
    print('\033[1;31mModel Contain : ', count_num_params(_p) / 1e9, ' Billion Parameters')


class JaxRNG(object):
    @classmethod
    def from_seed(cls, seed):
        """
            The from_seed function is a class method that takes a seed and returns an instance of the class.
            This allows us to create multiple instances of the same random number generator with different seeds,
            which can be useful for debugging or reproducibility.

            :param cls: Pass the class of the object that is being created
            :param seed: Initialize the random number generator
            :return: An instance of the class
            
            """

        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the random number generator, which will be used to generate
        random numbers for initializing weights and biases.

        :param self: Represent the instance of the class
        :param rng: Generate random numbers
        :return: The object itself
        
        """
        self.rng = rng

    def __call__(self, keys=None):
        """
        The __call__ function is a special function in Python that allows an object to be called like a function.

        :param self: Refer to the object itself
        :param keys: Split the random number generator into multiple parts
        :return: A random number generator
        
        """
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


def init_rng(seed):
    """
    The init_rng function initializes the global random number generator.

    :param seed: Initialize the random number generator
    :return: A random number generator
    
    """
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    """
    The next_rng function is a wrapper around jax.random.PRNGKey, which
    is used to generate random numbers in JAX. The next_rng function
    generates a new PRNGKey from the previous one, and updates the global
    variable jax_utils_rng with this new key.

    :param *args: Pass a variable number of arguments to the function
    :param **kwargs: Pass in a dictionary of parameters
    :return: A random number generator
    
    """
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


class GenerateRNG:
    def __init__(self, seed: int = 0):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the initial state of the object, which in this case includes a seed and a random number generator.
        The seed can be set by passing an argument to __init__, but if no argument is passed it defaults to 0.

        :param self: Represent the instance of the class
        :param seed: int: Set the seed for the random number generator
        :return: The object itself
        
        """
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)

    def __next__(self):
        """
        The __next__ function is called by the for loop to get the next value.
        It uses a while True loop so that it can return an infinite number of values.
        The function splits the random number generator into two parts, one part
        is used to generate a key and then returned, and the other part becomes
        the new random number generator.

        :param self: Represent the instance of the class
        :return: A random number
        
        """
        while True:
            self.rng, ke = jax.random.split(self.rng, 2)
            return ke
