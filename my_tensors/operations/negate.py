import numpy as np
from operation import Operation, Tensor
from typing import Tuple


class Negate(Operation):
    """Negate operation."""
    def _forward(self, input_1: Tensor) -> Tensor:
        """Forward pass of the operation."""
        # Perform the negate operation.
        output = Tensor(-input_1.data,
                        requires_grad=input_1.requires_grad,
                        grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of the operation."""
        grad_1 = -grad

        return (grad_1,)
