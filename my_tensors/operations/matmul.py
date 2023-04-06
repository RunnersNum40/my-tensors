from .operation import Operation, Tensor, Tuple, np


class MatMul(Operation):
    """Matrix multiplication operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def _forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of the matrix multiplication operation.

        Args:
            x (Tensor): First tensor.
            y (Tensor): Second tensor.

        Returns:
            Tensor: Matrix product of the two tensors.
        """
        data = np.matmul(x.data, y.data)
        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the matrix multiplication operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        # Gradient of the output with respect to the first input
        grad_x = np.matmul(grad, self.inputs[1].data.T)

        # Gradient of the output with respect to the second input
        grad_y = np.matmul(self.inputs[0].data.T, grad)

        return grad_x, grad_y
