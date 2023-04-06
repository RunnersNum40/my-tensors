import numpy as np
from operation import Operation, Tensor
from typing import Tuple


class Log(Operation):
    """Element-wise natural log operation."""
    def _forward(self, input_1: Tensor) -> Tensor:
        """Forward pass of the operation."""
        # Perform the log operation.
        output = Tensor(np.log(input_1.data),
                        requires_grad=input_1.requires_grad,
                        grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of the operation."""
        input_1 = self.inputs[0]

        grad_1 = 1 / input_1.data * grad

        return (grad_1,)
