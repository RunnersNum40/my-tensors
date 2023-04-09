from .operation import Operation, Tensor, Tuple, np


class Concatenate(Operation):
    """Concatenate operation.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def __init__(self, axis: int) -> None:
        super().__init__()

        self.axis = axis

    def _forward(self, *inputs: Tuple[Tensor, ...]) -> Tensor:
        """Forward pass of the operation.

        Args:
            inputs (Tuple[Tensor, ...]): Inputs to the operation.

        Returns:
            Tensor: Output of the operation.
        """
        self.inputs = inputs

        output = np.concatenate(inputs, axis=self.axis)

        return output

    def _backward(self, grad: Tensor) -> None:
        """Gradient of the operation.

        Args:
            grad (Tensor): Gradient of the output of the operation with
                respect to the output of the operation.
        """
        grads = []

        for input in self.inputs:
            shape = list(input.shape)
            shape[self.axis] = 1
            shape = tuple(shape)

            grads.append(grad.get_slice(self.axis, 0, shape))

        return tuple(grads)
