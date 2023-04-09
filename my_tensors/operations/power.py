from .operation import Operation, Tensor, Tuple, np


class Power(Operation):
    """Power operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def _forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of the power operation.

        Args:
            x (Tensor): Input tensor.
            y (Tensor)): Input tensor or scalar.

        Returns:
            Tensor: Power of the input tensor.
        """
        # Element-wise power if both inputs are tensors
        # Scalar power if one of the inputs is a scalar
        data = np.power(x.data, y.data)
        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the power operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        # Gradient of the output with respect to the first input
        grad_x = grad * self.inputs[1].data * np.power(self.inputs[0].data,
                                                       self.inputs[1].data - 1)
        if len(self.inputs[0].shape) == 0:
            grad_x = np.sum(grad_x)
        # Gradient of the output with respect to the second input
        grad_y = grad * np.power(self.inputs[0].data, self.inputs[1].data) * np.log(self.inputs[0].data)
        if len(self.inputs[1].shape) == 0:
            grad_y = np.sum(grad_y)

        return grad_x, grad_y
