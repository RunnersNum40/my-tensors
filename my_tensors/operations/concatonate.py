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
        # Get the numpy arrays from the inputs
        inputs_data = [input.data for input in inputs]
        data = np.concatenate(inputs_data, axis=self.axis)
        requires_grad = any(input.requires_grad for input in inputs)

        return Tensor(data=data, requires_grad=requires_grad, grad_fn=self)

    def _backward(self, grad: Tensor) -> None:
        """Gradient of the operation.

        Args:
            grad (Tensor): Gradient of the output of the operation with
                respect to the output of the operation.
        """
        grads = []

        for input in self.inputs:
            # Get the shape of the input
            shape = input.data.shape
            # Get the number of elements in the axis
            num_elements = shape[self.axis]
            # Remove the number of elements from the gradient along the axis
            grad_input = np.take(grad.data, range(num_elements), axis=self.axis)
            # Append the gradient to the list
            grads.append(grad_input)
            # Remove the number of elements from the gradient
            grad = np.delete(grad.data, range(num_elements), axis=self.axis)

        return tuple(grads)
