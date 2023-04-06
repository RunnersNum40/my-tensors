import numpy as np
from operation import Operation, Tensor
from typing import Tuple


class Maximum(Operation):
    """Maximum operation."""
    def __init__(self, axis) -> None:
        super().__init__()
        self.axis = axis

    def _forward(self, input_1: Tensor) -> Tensor:
        """Forward pass of the operation."""
        # Perform the maximum operation.
        output = Tensor(input_1.data.max(axis=self.axis),
                        requires_grad=input_1.requires_grad,
                        grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of the operation."""
        input_1 = self.inputs[0]

        grad_1 = np.zeros_like(input_1.data)

        return (grad_1,)
