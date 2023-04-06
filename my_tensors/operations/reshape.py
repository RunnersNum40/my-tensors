from .operation import Operation, Tensor, Tuple, np


class Reshape(Operation):
    """Reshape operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass of the reshape operation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Reshaped tensor.
        """
        data = x.data.reshape(self.shape)
        requires_grad = x.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the reshape operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        # Gradient of the output with respect to the input
        grad_x = grad.reshape(self.inputs[0].shape)

        return grad_x,
