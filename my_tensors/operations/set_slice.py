from .operation import Operation, Tensor, Tuple, Union, np


class SetSlice(Operation):
    """Set item operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def __init__(self, index: Union[int, slice, tuple]) -> None:
        super().__init__()
        self.index = index

    def _forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of the set item operation.

        Args:
            x (Tensor): Input tensor.
            y (Tensor): Value to set.

        Returns:
            Tensor: Tensor with the item set.
        """
        data = x.data.copy()
        data[self.index] = y.data
        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the set item operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        # Gradient of the output with respect to the first input
        grad_x = np.zeros_like(self.inputs[0].data)
        grad_x[self.index] = grad

        # Gradient of the output with respect to the second input
        grad_y = grad

        return grad_x, grad_y
