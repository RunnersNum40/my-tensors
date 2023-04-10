"""Tensor package.

Tensors are multidimensional arrays that can be used to represent data in
neural networks. They can also be used to represent the parameters of a
neural network.
"""

# Define the __all__ variable
__all__ = ["Tensor", "operations", "sum", "mean", "transpose", "matmul", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "exp", "log", "log2", "log10", "sqrt", "abs", "reshape", "flatten", "dot", "shape", "concatenate"]  # noqa: E501

# Import the tensor class
from my_tensors.tensor import Tensor
from my_tensors import operations

from my_tensors.tensor import sum
from my_tensors.tensor import mean
from my_tensors.tensor import transpose
from my_tensors.tensor import matmul
from my_tensors.tensor import sin
from my_tensors.tensor import cos
from my_tensors.tensor import tan
from my_tensors.tensor import arcsin
from my_tensors.tensor import arccos
from my_tensors.tensor import arctan
from my_tensors.tensor import sinh
from my_tensors.tensor import cosh
from my_tensors.tensor import tanh
from my_tensors.tensor import arcsinh
from my_tensors.tensor import arccosh
from my_tensors.tensor import arctanh
from my_tensors.tensor import exp
from my_tensors.tensor import log
from my_tensors.tensor import log2
from my_tensors.tensor import log10
from my_tensors.tensor import sqrt
from my_tensors.tensor import abs
from my_tensors.tensor import reshape
from my_tensors.tensor import flatten
from my_tensors.tensor import dot
from my_tensors.tensor import shape
from my_tensors.tensor import concatenate
