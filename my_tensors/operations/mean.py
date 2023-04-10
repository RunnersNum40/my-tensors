from .operation import Operation, Tensor, Tuple, np


class Mean(Operation):
    """Mean operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def __init__(self, axis: int = None) -> None:
        """Mean operation constructor.

        Args:
            axis (int, optional): Axis along which to mean. Defaults to None.
        """
        super().__init__()
        self.axis = axis

    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass of the mean operation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Mean of the tensor.
        """
        data = x.data.mean(axis=self.axis)
        requires_grad = x.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the mean operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        if self.axis is None:
            grad_x = np.ones_like(self.inputs[0].data) * grad
            grad_x /= self.inputs[0].data.size
        else:
            # Repeat the gradient along the axis
            grad_x = grad.reshape(1, *grad.shape) * np.ones_like(self.inputs[0].data)
            # Divide by the number of elements in the axis
            grad_x /= self.inputs[0].data.shape[self.axis]

        return grad_x,
