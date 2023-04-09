from .operation import Operation, Tensor, Tuple, np


class Divide(Operation):
    """Divide operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def _forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of the divide operation.

        Args:
            x (Tensor): First tensor.
            y (Tensor): Second tensor.

        Returns:
            Tensor: Quotient of the two tensors.
        """
        # Element-wise division if both inputs are tensors
        # Scalar division if one of the inputs is a scalar
        data = x.data / y.data
        requires_grad = x.requires_grad or y.requires_grad

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
        if len(self.inputs[0].shape) == 0:
            grad_x = np.sum(grad_x)
        # Gradient of the output with respect to the second input
        grad_y = -grad * self.inputs[0].data / self.inputs[1].data ** 2
        if len(self.inputs[1].shape) == 0:
            grad_y = np.sum(grad_y)

        return grad_x, grad_y
