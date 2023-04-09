"""Operations on tensors. Operations must always be performed on tensors.

Operations define the computational graph of a tensor. The computational graph
is used to calculate the gradient of the output of the graph with respect to
the initial parameters of the graph.

Attributes:
    inputs (Tuple[Tensor, ...]): Inputs to the operation.
    output (Tensor): Output of the operation.

Methods:
    forward: Forward pass of the operation.
    backward: Gradient of the operation.
"""

# Define the __all__ variable
__all__ = ["Operation", "Input", "Add", "Subtract", "Multiply", "Divide", "Sum", "Mean", "Transpose", "MatMul", "GetSlice", "SetSlice", "Reshape", "Flatten", "Sin", "Cos", "Tan", "ArcSin", "ArcCos", "ArcTan", "Sinh", "Cosh", "Tanh", "ArcSinh", "ArcCosh", "ArcTanh", "Exp", "Log", "Log2", "Log10", "Abs", "Sqrt", "Dot", "Concatenate", "Power"]  # noqa: E501

# Import the operations module
from my_tensors.operations.operation import Operation, Input
from my_tensors.operations.add import Add
from my_tensors.operations.subtract import Subtract
from my_tensors.operations.multiply import Multiply
from my_tensors.operations.divide import Divide
from my_tensors.operations.sum_ import Sum
from my_tensors.operations.mean import Mean
from my_tensors.operations.transpose import Transpose
from my_tensors.operations.matmul import MatMul
from my_tensors.operations.get_slice import GetSlice
from my_tensors.operations.set_slice import SetSlice
from my_tensors.operations.reshape import Reshape
from my_tensors.operations.flatten import Flatten
from my_tensors.operations.sin import Sin
from my_tensors.operations.cos import Cos
from my_tensors.operations.tan import Tan
from my_tensors.operations.arcsin import ArcSin
from my_tensors.operations.arccos import ArcCos
from my_tensors.operations.arctan import ArcTan
from my_tensors.operations.sinh import Sinh
from my_tensors.operations.cosh import Cosh
from my_tensors.operations.tanh import Tanh
from my_tensors.operations.arcsinh import ArcSinh
from my_tensors.operations.arccosh import ArcCosh
from my_tensors.operations.arctanh import ArcTanh
from my_tensors.operations.exp import Exp
from my_tensors.operations.log import Log
from my_tensors.operations.log2 import Log2
from my_tensors.operations.log10 import Log10
from my_tensors.operations.absolute import Abs
from my_tensors.operations.sqrt import Sqrt
from my_tensors.operations.dot import Dot
from my_tensors.operations.concatonate import Concatenate
from my_tensors.operations.power import Power
