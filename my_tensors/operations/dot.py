from .operation import Operation, Tensor, Tuple, Union, np


class Dot(Operation):
    """Dot operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def __init__(self, axes: Union[tuple[int, ...], int]) -> None:
        """Dot operation constructor.

        Args:
            axes (Union[tuple[int, ...], int]): Axes to sum over.
        """
        super().__init__()
        self.axes = axes

    def _forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Forward pass of the dot operation.

        Args:
            x (Tensor): First input tensor.
            y (Tensor): Second input tensor.

        Returns:
            Tensor: Dot product of the input tensors.
        """
        data = np.tensordot(x.data, y.data, axes=self.axes)
        requires_grad = x.requires_grad or y.requires_grad

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the dot operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.
        """
        # Gradient of the output with respect to the inputs
        grad_x = np.tensordot(grad, self.inputs[1].data, axes=self.axes)
        grad_y = np.tensordot(self.inputs[0].data, grad, axes=self.axes)

        return grad_x, grad_y
