"""Tensor package with automatic differentiation.

This package contains a tensor class that can be used to perform
automatic differentiation. The tensor class is a subclass of the
NumPy ndarray class, so it can be used in place of NumPy arrays.
"""

# Define the __all__ variable
__all__ = ['Tensor']

# Import the tensor class
from tensor import Tensor
