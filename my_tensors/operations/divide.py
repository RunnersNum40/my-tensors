from .operation import Operation, Tensor, Scalar, Tuple, Union, np


class Divide(Operation):
    """Divide operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def _forward(self, x: Tensor, y: Union[Tensor, Scalar]) -> Tensor:
        """Forward pass of the divide operation.

        Args:
            x (Tensor): First tensor.
            y (Tensor, Scalar): Second tensor.

        Returns:
            Tensor: Quotient of the two tensors.
        """
        if isinstance(y, x.__class__):
            # Element-wise division
            data = x.data / y.data
            requires_grad = x.requires_grad or y.requires_grad
        else:
            # Scalar division
            data = x.data / y
            requires_grad = x.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the divide operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        # Gradient of the output with respect to the first input
        grad_x = grad / self.inputs[1].data

        # Gradient of the output with respect to the second input
        if isinstance(self.inputs[1], self.inputs[0].__class__):
            grad_y = -grad * self.inputs[0].data / self.inputs[1].data ** 2
        else:
            grad_y = None

        return grad_x, grad_y
