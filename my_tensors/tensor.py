import numpy as np
from typing import Union, Tuple, List


class Tensor:
    """Tensor class.

    Tensors are used to represent data in neural networks. They are used to
    represent the input to the neural network, the output of the neural
    network, and the parameters of the neural network.

    A tensor tracks the graph of computations that produced it. This allows the
    gradient of the tensor with respect to inital parameters to be calculated
    using the chain rule.

    Attributes:
        data (np.ndarray): Data in the tensor.
        grad (np.ndarray): Gradient of the tensor with respect to the inital
            parameters.
        grad_fn (Operation): Operation that created the tensor.
        requires_grad (bool): Whether the gradient of the tensor with respect
            to the inital parameters should be calculated.
    """
    def __init__(self,
                 data: Union[np.ndarray, list, tuple, int, float],
                 requires_grad: bool = True,
                 grad_fn: "operations.Operation" = None
                 ) -> None:
        """Initializes a tensor.

        Args:
            data (np.ndarray): Data in the tensor.
            requires_grad (bool): Whether the gradient of the tensor with
                respect to the inital parameters should be calculated.
            grad_fn (Operation): Operation that created the tensor.
        """
        # Save the input data
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        # Initialize the gradient
        self.zero_grad()

    def zero_grad(self) -> None:
        """Sets the gradient of the tensor to None."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def backward(self,
                 grad: np.ndarray = None
                 ) -> Union[Tuple[np.ndarray, ...], None]:
        """Performs a backward pass through the computational graph.

        Args:
            grad (np.ndarray): Gradient of the output of the computational
                graph with respect to the output of the current tensor.
        """
        # Cut off the computational graph if the tensor does not require a
        # gradient
        if self.requires_grad:
            # If the gradient is not provided, initialize it to 1
            if grad is None:
                grad = np.ones_like(self.data)
            # Make sure the gradient has at least one dimension
            if grad.ndim == 0:
                grad = grad.reshape(1)
            # If the tensor is not a leaf node
            if not self.leaf:
                grads = self.grad_fn.backward(grad)
                for input_tensor, input_grad in zip(self.grad_fn.inputs, grads):  # noqa: E501
                    if isinstance(input_tensor, Tensor):
                        input_tensor.backward(input_grad)
            # If the tensor is a leaf node
            else:
                self.grad += grad

    @property
    def leaf(self) -> bool:
        """Returns whether the tensor is a leaf node in the computational
        graph.

        Nodes that are leaf nodes in the computational graph are the input
        tensors to the computational graph.
        """
        return self.grad_fn is None

    @property
    def shape(self) -> tuple:
        """Returns the shape of the tensor."""
        return self.data.shape

    @property
    def T(self) -> "Tensor":
        """Returns the transpose of the tensor."""
        return self.transpose()

    def detach(self) -> "Tensor":
        """Returns a tensor with the same data as the current tensor but with
        requires_grad set to False.
        """
        return Tensor(self.data, False)

    def numpy(self) -> np.ndarray:
        """Returns the data in the tensor as a numpy array."""
        return self.data

    def list(self) -> list:
        """Returns the data in the tensor as a list."""
        return self.data.tolist()

    def __add__(self, other: "operations.Input") -> "Tensor":
        """Adds a tensor to the current tensor."""
        operation = operations.Add()
        return operation(self, other)

    def __radd__(self, other: "operations.Input") -> "Tensor":
        """Adds a tensor to the current tensor."""
        operation = operations.Add()
        return operation(self, other)

    def __sub__(self, other: "operations.Input") -> "Tensor":
        """Subtracts a tensor from the current tensor."""
        operation = operations.Subtract()
        return operation(self, other)

    def __rsub__(self, other: "operations.Input") -> "Tensor":
        """Subtracts the current tensor from a tensor."""
        operation = operations.Subtract()
        return operation(other, self)

    def __mul__(self, other: "operations.Input") -> "Tensor":
        """Multiplies a tensor by the current tensor."""
        operation = operations.Multiply()
        return operation(self, other)

    def __rmul__(self, other: "operations.Input") -> "Tensor":
        """Multiplies a tensor by the current tensor."""
        operation = operations.Multiply()
        return operation(self, other)

    def __truediv__(self, other: "operations.Input") -> "Tensor":
        """Divides a tensor by the current tensor."""
        operation = operations.Divide()
        return operation(self, other)

    def __rtruediv__(self, other: "operations.Input") -> "Tensor":
        """Divides a tensor by the current tensor."""
        operation = operations.Divide()
        return operation(other, self)

    def __pow__(self, other: "operations.Input") -> "Tensor":
        """Raises a tensor to the power of the current tensor."""
        operation = operations.Power()
        return operation(self, other)

    def __rpow__(self, other: "operations.Input") -> "Tensor":
        """Raises a tensor to the power of the current tensor."""
        operation = operations.Power()
        return operation(other, self)

    def __neg__(self) -> "Tensor":
        """Negates the current tensor."""
        operation = operations.Negate()
        return operation(self)

    def __abs__(self) -> "Tensor":
        """Returns the absolute value of the current tensor."""
        operation = operations.Abs()
        return operation(self)

    def __matmul__(self, other: "operations.Input") -> "Tensor":
        """Performs a matrix multiplication between a tensor and the current
        tensor.
        """
        operation = operations.MatMul()
        return operation(self, other)

    def __rmatmul__(self, other: "operations.Input") -> "Tensor":
        """Performs a matrix multiplication between a tensor and the current
        tensor.
        """
        operation = operations.MatMul()
        return operation(other, self)

    def __getitem__(self, key: Union[int, slice, tuple]) -> "Tensor":
        """Returns a slice of the tensor."""
        operation = operations.GetSlice(key)
        return operation(self)

    def __setitem__(self,
                    key: Union[int, slice, tuple],
                    value: "Tensor") -> None:
        """Sets a slice of the tensor."""
        operation = operations.SetSlice(key)
        operation(self, value)

    def matmul(self, other: "operations.Input") -> "Tensor":
        """Alias for the matmul function."""
        return matmul(self, other)

    def sum(self, axis: int = None) -> "Tensor":
        """Alias for the sum function."""
        return sum(self, axis)

    def mean(self, axis: int = None) -> "Tensor":
        """Alias for the mean function."""
        return mean(self, axis)

    def transpose(self) -> "Tensor":
        """Alias for the transpose function."""
        return transpose(self)

    def sin(self) -> "Tensor":
        """Alias for the sin function."""
        return sin(self)

    def cos(self) -> "Tensor":
        """Alias for the cos function."""
        return cos(self)

    def tan(self) -> "Tensor":
        """Alias for the tan function."""
        return tan(self)

    def arcsin(self) -> "Tensor":
        """Alias for the arcsin function."""
        return arcsin(self)

    def arccos(self) -> "Tensor":
        """Alias for the arccos function."""
        return arccos(self)

    def arctan(self) -> "Tensor":
        """Alias for the arctan function."""
        return arctan(self)

    def sinh(self) -> "Tensor":
        """Alias for the sinh function."""
        return sinh(self)

    def cosh(self) -> "Tensor":
        """Alias for the cosh function."""
        return cosh(self)

    def tanh(self) -> "Tensor":
        """Alias for the tanh function."""
        return tanh(self)

    def arcsinh(self) -> "Tensor":
        """Alias for the arcsinh function."""
        return arcsinh(self)

    def arccosh(self) -> "Tensor":
        """Alias for the arccosh function."""
        return arccosh(self)

    def arctanh(self) -> "Tensor":
        """Alias for the arctanh function."""
        return arctanh(self)

    def exp(self) -> "Tensor":
        """Alias for the exp function."""
        return exp(self)

    def log(self) -> "Tensor":
        """Alias for the log function."""
        return log(self)

    def log2(self) -> "Tensor":
        """Alias for the log2 function."""
        return log2(self)

    def log10(self) -> "Tensor":
        """Alias for the log10 function."""
        return log10(self)

    def sqrt(self) -> "Tensor":
        """Alias for the sqrt function."""
        return sqrt(self)

    def reshape(self, shape: tuple) -> "Tensor":
        """Alias for the reshape function."""
        return reshape(self, shape)

    def flatten(self) -> "Tensor":
        """Alias for the flatten function."""
        return flatten(self)

    def graph(self, level: int = 0) -> str:
        """Returns a string representation of the computational graph.

        Args:
            level: The level of the computational graph the tensor is on.
        """
        indent = "\t" * level
        if self.grad_fn is not None:
            ret = f"{indent}{self.grad_fn}\n"
            for child in self.grad_fn.inputs:
                if isinstance(child, Tensor):
                    ret += child.graph(level + 1)
        else:
            ret = f"{indent}Parameter({self.shape})\n"

        return ret

    def __len__(self) -> int:
        """Returns the length of the tensor."""
        return len(self.data)

    def __str__(self) -> str:
        """Returns a string representation of the tensor."""
        return f"Tensor({(self.data).__str__()})"

    def __repr__(self) -> str:
        """Returns a string representation of the tensor."""
        return f"Tensor({(self.data).__repr__()})"


def sum(tensor: Tensor, axis: int = None) -> Tensor:
    """Returns the sum of a tensor."""
    operation = operations.Sum(axis)
    return operation(tensor)


def mean(tensor: Tensor, axis: int = None) -> Tensor:
    """Returns the mean of a tensor."""
    operation = operations.Mean(axis)
    return operation(tensor)


def transpose(tensor: Tensor) -> Tensor:
    """Returns the transpose of a tensor."""
    operation = operations.Transpose()
    return operation(tensor)


def matmul(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Performs a matrix multiplication between two tensors."""
    operation = operations.MatMul()
    return operation(tensor1, tensor2)


def sin(tensor: Tensor) -> Tensor:
    """Returns the sine of a tensor."""
    operation = operations.Sin()
    return operation(tensor)


def cos(tensor: Tensor) -> Tensor:
    """Returns the cosine of a tensor."""
    operation = operations.Cos()
    return operation(tensor)


def tan(tensor: Tensor) -> Tensor:
    """Returns the tangent of a tensor."""
    operation = operations.Tan()
    return operation(tensor)


def arcsin(tensor: Tensor) -> Tensor:
    """Returns the arcsine of a tensor."""
    operation = operations.Arcsin()
    return operation(tensor)


def arccos(tensor: Tensor) -> Tensor:
    """Returns the arccosine of a tensor."""
    operation = operations.Arccos()
    return operation(tensor)


def arctan(tensor: Tensor) -> Tensor:
    """Returns the arctangent of a tensor."""
    operation = operations.Arctan()
    return operation(tensor)


def sinh(tensor: Tensor) -> Tensor:
    """Returns the hyperbolic sine of a tensor."""
    operation = operations.Sinh()
    return operation(tensor)


def cosh(tensor: Tensor) -> Tensor:
    """Returns the hyperbolic cosine of a tensor."""
    operation = operations.Cosh()
    return operation(tensor)


def tanh(tensor: Tensor) -> Tensor:
    """Returns the hyperbolic tangent of a tensor."""
    operation = operations.Tanh()
    return operation(tensor)


def arcsinh(tensor: Tensor) -> Tensor:
    """Returns the hyperbolic arcsine of a tensor."""
    operation = operations.Arcsinh()
    return operation(tensor)


def arccosh(tensor: Tensor) -> Tensor:
    """Returns the hyperbolic arccosine of a tensor."""
    operation = operations.Arccosh()
    return operation(tensor)


def arctanh(tensor: Tensor) -> Tensor:
    """Returns the hyperbolic arctangent of a tensor."""
    operation = operations.Arctanh()
    return operation(tensor)


def exp(tensor: Tensor) -> Tensor:
    """Returns the exponential of a tensor."""
    operation = operations.Exp()
    return operation(tensor)


def log(tensor: Tensor) -> Tensor:
    """Returns the natural logarithm of a tensor."""
    operation = operations.Log()
    return operation(tensor)


def log2(tensor: Tensor) -> Tensor:
    """Returns the base 2 logarithm of a tensor."""
    operation = operations.Log2()
    return operation(tensor)


def log10(tensor: Tensor) -> Tensor:
    """Returns the base 10 logarithm of a tensor."""
    operation = operations.Log10()
    return operation(tensor)


def sqrt(tensor: Tensor) -> Tensor:
    """Returns the square root of a tensor."""
    operation = operations.Sqrt()
    return operation(tensor)


def abs(tensor: Tensor) -> Tensor:
    """Returns the absolute value of a tensor."""
    operation = operations.Abs()
    return operation(tensor)


def reshape(tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshapes a tensor."""
    operation = operations.Reshape(shape)
    return operation(tensor)


def flatten(tensor: Tensor) -> Tensor:
    """Flattens a tensor."""
    operation = operations.Flatten()
    return operation(tensor)


def dot(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """Performs a dot product between two tensors."""
    operation = operations.Dot()
    return operation(tensor1, tensor2)


def shape(tensor: Tensor) -> Tuple[int, ...]:
    """Alias for the shape property of a tensor."""
    return tensor.shape


def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Concatenates a list of tensors."""
    operation = operations.Concatenate(axis)
    return operation(tensors)


# This deals with the circular import
import my_tensors.operations as operations  # noqa: E402
