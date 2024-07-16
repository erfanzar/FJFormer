class SymbolicConstantError(Exception):
    """Base exception class for SymbolicConstant-related errors."""

    pass


class ShapeDtypeError(SymbolicConstantError):
    """Exception raised for errors in shape or dtype determination."""

    pass


class OperationError(SymbolicConstantError):
    """Exception raised when an operation on SymbolicConstant fails."""

    pass


class MaterializationError(SymbolicConstantError):
    """Exception raised when materialization of a SymbolicConstant fails."""

    pass


class UnsupportedPrimitiveError(SymbolicConstantError):
    """Exception raised when an unsupported JAX primitive is encountered."""

    pass


class UninitializedAval(Exception):
    """Exception raised when an aval is accessed before initialization."""

    def __init__(self, kind):
        super().__init__(
            (
                f"{kind} was not set during initialization. Shape and dtype may be set by:"
                "\n\t1. Directly passing them as keyword arguments to ImplicitArray instances"
                "\n\t2. Overriding the default_shape/default_dtype class attributes"
                "\n\t3. Overriding the compute_shape/compute_dtype methods"
                "\n\t4. Overriding __post_init__ and setting their values there"
                "\n\t5. None of the above, in which case `materialize()` will be called in an attempt to infer them."
                " If their values are required in order to compute the materialization this will be unsuccessful."
            )
        )


class ImplicitArrayError(Exception):
    """Base exception for ImplicitArray-related errors."""

    pass
