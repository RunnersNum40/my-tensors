import numpy as np
from operation import Operation, Tensor
from typing import Tuple


class Sum(Operation):
    """Sum operation."""
    def __init__(self, axis: int = None) -> None:
        super().__init__()
        self.axis = axis

    def _forward(self, input_1: Tensor) -> Tensor:
        """Forward pass of the operation."""
        # Perform the sum operation.
        require_grad = input_1.requires_grad
        output = Tensor(input_1.data.sum(axis=self.axis),
                        requires_grad=require_grad,
                        grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of the operation."""
        input_1 = self.inputs[0]

        if self.axis is None:
            # Multiply the gradient along the sum axis by the input.
            grad_1 = input_1.data * grad
        else:
            # Multiply the gradient along the sum axis by the input.
            grad_1 = np.swapaxes(np.swapaxes(input_1.data,
                                             self.axis,
                                             -1) * grad,
                                 -1,
                                 self.axis)

        return (grad_1,)
