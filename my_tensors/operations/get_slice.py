from .operation import Operation, Tensor, Tuple, Union, np


class GetSlice(Operation):
    """Get slice operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def __init__(self, key: Union[int, slice, tuple]) -> None:
        super().__init__()
        self.key = key

    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass of the get slice operation.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Sliced tensor.
        """
        data = x.data[self.key]
        requires_grad = x.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the get slice operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        # Gradient of the output with respect to the input
        grad_x = np.zeros_like(self.inputs[0].data)
        grad_x[self.key] = grad

        return grad_x,
