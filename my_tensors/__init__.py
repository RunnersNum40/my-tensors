"""Tensor package.

Tensors are multidimensional arrays that can be used to represent data in
neural networks. They can also be used to represent the parameters of a
neural network.
"""

# Define the __all__ variable
__all__ = ["Tensor", "operations"]

# Import the tensor class
from my_tensors.tensor import Tensor
from my_tensors import operations
