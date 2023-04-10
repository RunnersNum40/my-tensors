from .operation import Operation, Tensor, Tuple, np


class Sum(Operation):
    """Sum operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def __init__(self, axis: int = None) -> None:
        """Sum operation constructor.

        Args:
            axis (int, optional): Axis along which to sum. Defaults to None.
        """
        super().__init__()
        self.axis = axis

    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass of the sum operation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Sum of the tensor.
        """
        data = x.data.sum(axis=self.axis)
        requires_grad = x.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the sum operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        if self.axis is None:
            grad_x = np.ones_like(self.inputs[0].data) * grad
        else:
            # Repeat the gradient along the axis
            grad_x = grad.reshape(1, *grad.shape) * np.ones_like(self.inputs[0].data)

        return grad_x,
