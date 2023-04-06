from .operation import Operation, Tensor, Tuple, np


class Log10(Operation):
    """Log10 operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass of the log10 operation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Log10 of the input tensor.
        """
        data = np.log10(x.data)
        requires_grad = x.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the log10 operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        # Gradient of the output with respect to the input
        grad_x = grad / (self.inputs[0].data * np.log(10))

        return grad_x,
