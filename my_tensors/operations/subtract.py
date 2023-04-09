from .operation import Operation, Tensor, Tuple, np


class Subtract(Operation):
    """Subtract operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def _forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of the subtract operation.

        Args:
            x (Tensor): First tensor.
            y (Tensor, Scalar): Second tensor.

        Returns:
            Tensor: Difference of the two tensors.
        """
        # Element-wise subtraction if both inputs are tensors
        # Scalar subtraction if one of the inputs is a scalar
        data = x.data - y.data
        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the subtract operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        # Gradient of the output with respect to the first input
        grad_x = grad
        if len(self.inputs[0].shape) == 0:
            grad_x = np.sum(grad_x)
        # Gradient of the output with respect to the second input
        grad_y = -grad
        if len(self.inputs[1].shape) == 0:
            grad_y = -np.sum(grad_y)

        return grad_x, grad_y
