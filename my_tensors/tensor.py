import numpy as np
from typing import Union

import operations as operations


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
    def __init__(self, data: Union[np.ndarray, list, tuple],
                 requires_grad: bool = False,
                 grad_fn: operations.Operation = None
                 ) -> None:
        """Constructor.

        Args:
            data (np.ndarray): Data in the tensor.
            requires_grad (bool): Whether the gradient of the tensor with
                respect to the inital parameters should be calculated.
        """
        # Save the data and whether the gradient should be calculated.
        self.data = np.array(data)
        self.requires_grad = requires_grad
        # Save the operation that created the tensor.
        self.grad_fn = grad_fn
        # Initialize the gradient to None.
        self.grad = None

    def detach(self) -> np.ndarray:
        """Detach the tensor from the graph of computations.

        Returns:
            np.ndarray: Data in the tensor.
        """
        # Return the data in the tensor.
        return self.data

    def backward(self, grad: np.ndarray = None) -> None:
        """Calculate the gradient of the tensor with respect to the inital
        parameters.

        Args:
            grad (np.ndarray): Gradient of the loss with respect to the tensor.
        """
        if self.requires_grad:
            if grad is None:
                # If the gradient of the loss with respect to the tensor has
                # not been calculated.
                grad = np.ones_like(self.data)

            if self.grad_fn:
                grads = self.grad_fn.backward(grad)
                for input_grad, input_ in zip(grads, self.grad_fn.inputs):
                    if input_grad is not None and isinstance(input_, Tensor):
                        input_.backward(input_grad)
            else:
                # Tensor is a parameter.
                self.grad = grad

    def sum(self, axis: int = None) -> 'Tensor':
        """Sum the tensor.

        Args:
            axis (int): Axis to sum over.

        Returns:
            Tensor: Summed tensor.
        """
        operation = operations.Sum(axis)
        return operation(self)

    def transpose(self, axes: list = None) -> 'Tensor':
        """Transpose the tensor.

        Args:
            axes (list): Axes to transpose over.

        Returns:
            Tensor: Transposed tensor.
        """
        operation = operations.Transpose(axes)
        return operation(self)

    @property
    def T(self) -> 'Tensor':
        """Transpose the tensor.

        Returns:
            Tensor: Transposed tensor.
        """
        return self.transpose()

    def reshape(self, shape: list) -> 'Tensor':
        """Reshape the tensor.

        Args:
            shape (list): Shape to reshape the tensor to.

        Returns:
            Tensor: Reshaped tensor.
        """
        operation = operations.Reshape(shape)
        return operation(self)

    def dot(self, other: 'Tensor') -> 'Tensor':
        """Dot product of the tensor.

        Args:
            other (Tensor): Tensor to dot product the tensor by.

        Returns:
            Tensor: Dot product of the tensor.
        """
        operation = operations.Dot()
        return operation(self, other)

    def log(self) -> 'Tensor':
        """Log of the tensor.

        Returns:
            Tensor: Log of the tensor.
        """
        operation = operations.Log()
        return operation(self)

    def min(self, axis: int = None) -> 'Tensor':
        """Minimum of the tensor.

        Args:
            axis (int): Axis to find the minimum over.

        Returns:
            Tensor: Minimum of the tensor.
        """
        operation = operations.Minimum(axis)
        return operation(self)

    def max(self, axis: int = None) -> 'Tensor':
        """Maximum of the tensor.

        Args:
            axis (int): Axis to find the maximum over.

        Returns:
            Tensor: Maximum of the tensor.
        """
        operation = operations.Maximum(axis)
        return operation(self)

    def __add__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Addition operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to add to the tensor.

        Returns:
            Tensor: Result of the addition.
        """
        opperation = operations.Add()
        return opperation(self, other)

    def __radd__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Addition operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to add to the tensor.

        Returns:
            Tensor: Result of the addition.
        """
        return self + other

    def __mul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Multiplication operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to multiply the tensor
                by.

        Returns:
            Tensor: Result of the multiplication.
        """
        operation = operations.Multiply()
        return operation(self, other)

    def __rmul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Multiplication operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to multiply the tensor
                by.

        Returns:
            Tensor: Result of the multiplication.
        """
        return self * other

    def __sub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Subtraction operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to subtract from the
                tensor.

        Returns:
            Tensor: Result of the subtraction.
        """
        operation = operations.Subtract()
        return operation(self, other)

    def __rsub__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Subtraction operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to subtract from the
                tensor.

        Returns:
            Tensor: Result of the subtraction.
        """
        operation = operations.Subtract()
        return operation(other, self)

    def __truediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Division operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to divide the tensor
                by.

        Returns:
            Tensor: Result of the division.
        """
        operation = operations.Divide()
        return operation(self, other)

    def __rtruediv__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Division operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to divide the tensor
                by.

        Returns:
            Tensor: Result of the division.
        """
        operation = operations.Divide()
        return operation(other, self)

    def __matmul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Matrix multiplication operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to multiply the tensor
                by.

        Returns:
            Tensor: Result of the matrix multiplication.
        """
        operation = operations.Dot()
        return operation(self, other)

    def __rmatmul__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Matrix multiplication operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to multiply the tensor
                by.

        Returns:
            Tensor: Result of the matrix multiplication.
        """
        return self @ other

    def __pow__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Exponentiation operator.

        Args:
            other (Union['Tensor', int, float]): Exponent to raise the tensor
                to.

        Returns:
            Tensor: Result of the exponentiation.
        """
        operation = operations.Power()
        return operation(self, other)

    def __neg__(self) -> 'Tensor':
        """Negation operator.

        Returns:
            Tensor: Result of the negation.
        """
        operation = operations.Negate()
        return operation(self)

    def __getitem__(self, index: Union[int, slice]) -> 'Tensor':
        """Indexing operator.

        Args:
            index (Union[int, slice]): Index to access.

        Returns:
            Tensor: Result of the indexing.
        """
        operation = operations.Index()
        return operation(self, index)

    def __setitem__(self,
                    index: Union[int, slice],
                    value: Union['Tensor', int, float]
                    ) -> None:
        """Indexing operator.

        Args:
            index (Union[int, slice]): Index to access.
            value (Union['Tensor', int, float]): Value to set the index to.
        """
        operation = operations.Index()
        operation(self, index, value)

    def __len__(self) -> int:
        """Length operator.

        Returns:
            int: Length of the tensor.
        """
        return len(self.data)

    def __iter__(self) -> 'Tensor':
        """Iterator operator.

        Returns:
            Tensor: Tensor to iterate over.
        """
        return self

    def __next__(self) -> 'Tensor':
        """Next operator.

        Returns:
            Tensor: Next tensor in the iteration.
        """
        if self.index < len(self.data):
            result = self[self.index]
            self.index += 1
            return result
        else:
            self.index = 0
            raise StopIteration

    def __eq__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Equality operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to compare the tensor
                to.

        Returns:
            Tensor: Result of the comparison.
        """
        operation = operations.Equal()
        return operation(self, other)

    def __ne__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Inequality operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to compare the tensor
                to.

        Returns:
            Tensor: Result of the comparison.
        """
        operation = operations.NotEqual()
        return operation(self, other)

    def __lt__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Less than operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to compare the tensor
                to.

        Returns:
            Tensor: Result of the comparison.
        """
        operation = operations.LessThan()
        return operation(self, other)

    def __le__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Less than or equal to operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to compare the tensor
                to.

        Returns:
            Tensor: Result of the comparison.
        """
        operation = operations.LessThanOrEqual()
        return operation(self, other)

    def __gt__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Greater than operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to compare the tensor
                to.

        Returns:
            Tensor: Result of the comparison.
        """
        operation = operations.GreaterThan()
        return operation(self, other)

    def __ge__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Greater than or equal to operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to compare the tensor
                to.

        Returns:
            Tensor: Result of the comparison.
        """
        operation = operations.GreaterThanOrEqual()
        return operation(self, other)

    def __and__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """And operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to compare the tensor
                to.

        Returns:
            Tensor: Result of the comparison.
        """
        operation = operations.And()
        return operation(self, other)

    def __or__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Or operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to compare the tensor
                to.

        Returns:
            Tensor: Result of the comparison.
        """
        operation = operations.Or()
        return operation(self, other)

    def __xor__(self, other: Union['Tensor', int, float]) -> 'Tensor':
        """Xor operator.

        Args:
            other (Union['Tensor', int, float]): Tensor to compare the tensor
                to.

        Returns:
            Tensor: Result of the comparison.
        """
        operation = operations.Xor()
        return operation(self, other)

    def __abs__(self) -> 'Tensor':
        """Absolute value operator.

        Returns:
            Tensor: Result of the absolute value.
        """
        operation = operations.Absolute()
        return operation(self)

    def __invert__(self) -> 'Tensor':
        """Inversion operator.

        Returns:
            Tensor: Result of the inversion.
        """
        operation = operations.Not()
        return operation(self)

    def __repr__(self) -> str:
        """Representation of the tensor.

        Returns:
            str: Representation of the tensor.
        """
        return f"Tensor({self.data})"
