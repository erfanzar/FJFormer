import warnings

import jax
from jax._src.core import Trace
from jax.core import get_aval, Tracer, full_lower

from .implicit_array import ImplicitArray


class ImplicitArrayTracer(Tracer):
    def __init__(self, trace, value):

        """
        The __init__ function is called when the class is instantiated.
        It sets up the object with all of its properties and methods.
        The self parameter refers to the instance of the object itself.

        :param self: Refer to the instance of the class
        :param trace: Store the traceback object, which is used to print out a stack trace
        :param value: Store the value of the exception
        :return: The value of the class
        """
        super().__init__(trace)
        self.value = value

    @property
    def aval(self):

        """
        The aval function is used to determine the shape and dtype of a value.

        :param self: Refer to the object itself
        :return: The aval of the value
        
        """
        if isinstance(self.value, ImplicitArray):
            return jax.ShapedArray(self.value.shape, self.value.dtype)
        return get_aval(self.value)

    def full_lower(self):

        """
        The full_lower function is used to convert an expression into a form that can be
           evaluated by the SymPy lambdify function.  The full_lower function will recursively
           descend through the expression tree and replace any instances of ImplicitArray with
           their value attribute.  This allows for expressions like:

        :param self: Refer to the current object
        :return: An implicitarray object
        """
        if isinstance(self.value, ImplicitArray):
            return self

        return full_lower(self.value)


class ImplicitArrayTrace(Trace):
    pure = lift = lambda self, val: ImplicitArrayTracer(self, val)

    def process_primitive(self, primitive, tracers, params):

        """
        The process_primitive function is called by the tracer when it encounters a primitive.
        The function should return a list of Tracers, which will be used to replace the original
        Tracers in the trace. The process_primitive function can also modify params, which are
        the parameters passed to the primitive.

        :param self: Access the class attributes
        :param primitive: Identify the primitive operation
        :param tracers: Trace the value of each input to a primitive
        :param params: Pass in the parameters of the function
        :return: The primitive, tracers and params

        """
        vals = [t.value for t in tracers]
        n_implicit = sum(isinstance(v, ImplicitArray) for v in vals)
        assert 1 <= n_implicit <= 2
        if n_implicit == 2:
            warnings.warn(f'Encountered op {primitive.name} with two implicit inputs so second will be materialized.')
            vals[1] = vals[1].materialize()
