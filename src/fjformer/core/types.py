from typing import Any, Optional, Tuple, Type, TypeVar

from plum import dispatch, parametric


# Custom exceptions
class ComplementError(Exception):
    """Base exception class for Complement-related errors."""

    pass


class TypeParameterError(ComplementError):
    """Exception raised for errors in type parameter initialization or comparison."""

    pass


# Type variables for better type hinting
A = TypeVar("A")
B = TypeVar("B")


class _ComplementMeta(type):
    def __instancecheck__(self, x: Any) -> bool:
        """
        Check if an object is an instance of the Complement type.

        Args:
            x: The object to check.

        Returns:
            bool: True if x is an instance of Complement, False otherwise.
        """
        try:
            a, b = self.type_parameter
            return a is None or (isinstance(x, a) and not isinstance(x, b))
        except Exception as e:
            raise TypeParameterError(f"Error in instance check: {str(e)}")


@parametric
class Complement(metaclass=_ComplementMeta):
    """
    Represents the relative complement of two types.

    The Complement[A, B] represents all elements in A that are not in B (A - B).

    Attributes:
        type_parameter (Tuple[Optional[Type], Optional[Type]]): A tuple containing the two types A and B.
    """

    @classmethod
    @dispatch
    def __init_type_parameter__(
        cls,
        a: Optional[Type[A]],
        b: Optional[Type[B]],
    ) -> Tuple[Optional[Type[A]], Optional[Type[B]]]:
        """
        Initialize the type parameters for the Complement.

        Args:
            a: The first type (superset).
            b: The second type (subset to be removed from a).

        Returns:
            A tuple containing the two type parameters.

        Raises:
            TypeParameterError: If there's an error initializing the type parameters.
        """
        try:
            return a, b
        except Exception as e:
            raise TypeParameterError(f"Error initializing type parameters: {str(e)}")

    @classmethod
    @dispatch
    def __le_type_parameter__(
        cls,
        left: Tuple[Optional[Type[A]], Optional[Type[B]]],
        right: Tuple[Optional[Type[A]], Optional[Type[B]]],
    ) -> bool:
        """
        Compare two Complement type parameters for the less than or equal relationship.

        This method defines the subtyping relationship between two Complement types.

        Args:
            left: The left-hand side Complement type parameters.
            right: The right-hand side Complement type parameters.

        Returns:
            bool: True if the left Complement is a subtype of the right Complement, False otherwise.

        Raises:
            TypeParameterError: If there's an error comparing the type parameters.
        """
        try:
            a_left, b_left = left
            a_right, b_right = right

            # Check if a_left is a subclass of a_right and b_right is a subclass of b_left
            is_subclass_a = issubclass(a_left, a_right) if a_left and a_right else True
            is_subclass_b = issubclass(b_right, b_left) if b_right and b_left else True

            return is_subclass_a and is_subclass_b
        except Exception as e:
            raise TypeParameterError(f"Error comparing type parameters: {str(e)}")

    @classmethod
    def is_complement(cls, x: Any) -> bool:
        """
        Check if an object is an instance of any Complement type.

        Args:
            x: The object to check.

        Returns:
            bool: True if x is an instance of any Complement type, False otherwise.
        """
        return isinstance(x, Complement)
