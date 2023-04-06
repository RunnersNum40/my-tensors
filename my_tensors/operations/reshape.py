import numpy as np
from operation import Operation, Tensor
from typing import Tuple


class Reshape(Operation):
    """Reshape operation."""
    def __init__(self, shape: tuple):
        """Initialize the operation."""
        super().__init__()
        self.shape = shape

    def _forward(self, input_1: Tensor) -> Tensor:
        """Forward pass of the operation."""
        require_grad = input_1.requires_grad
        output = Tensor(input_1.data.reshape(self.shape),
                        requires_grad=require_grad,
                        grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of the operation."""
        input_1 = self.inputs[0]

        grad_1 = grad.reshape(input_1.data.shape)

        return (grad_1,)
