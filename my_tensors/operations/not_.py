import numpy as np
from operation import Operation, Tensor
from typing import Tuple


class Not(Operation):
    """Not operation."""
    def _forward(self, input_1: Tensor) -> Tensor:
        """Forward pass of the operation."""
        # Perform the not operation.
        output = Tensor(~input_1.data,
                        requires_grad=input_1.requires_grad,
                        grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of the operation."""
        input_1 = self.inputs[0]

        grad_1 = np.zeros_like(input_1.data)

        return (grad_1,)
