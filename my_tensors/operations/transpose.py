import numpy as np
from operation import Operation, Tensor
from typing import Tuple


class Transpose(Operation):
    """Transpose operation."""
    def _forward(self, input_1: Tensor) -> Tensor:
        """Forward pass of the operation."""
        require_grad = input_1.requires_grad
        output = Tensor(input_1.data.T,
                        requires_grad=require_grad,
                        grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of the operation."""
        grad_1 = grad.T

        return (grad_1,)
