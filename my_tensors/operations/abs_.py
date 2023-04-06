import numpy as np
from operation import Operation, Tensor
from typing import Tuple


class Abs(Operation):
    """Absolute value operation."""
    def _forward(self, input_: Tensor) -> Tensor:
        """Forward pass of the operation."""
        # Perform the absolute value operation.
        require_grad = input_.requires_grad
        output = Tensor(np.abs(input_.data),
                        requires_grad=require_grad,
                        grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of the operation."""
        input_1 = self.inputs[0]
        grad_1 = np.zeros_like(self.input_1.data)
        grad_1[input_1.data >= 0] = 1
        grad_1[input_1.data < 0] = -1

        return (grad_1,)
