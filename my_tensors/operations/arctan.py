from .operation import Operation, Tensor, Tuple, np


class ArcTan(Operation):
    """ArcTan operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass of the arctan operation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: ArcTan of the input tensor.
        """
        data = np.arctan(x.data)
        requires_grad = x.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the arctan operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        # Gradient of the output with respect to the input
        grad_x = grad / (1 + self.inputs[0].data ** 2)

        return grad_x,
