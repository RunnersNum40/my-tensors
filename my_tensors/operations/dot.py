import numpy as np
from operation import Operation, Tensor
from typing import Tuple


class Dot(Operation):
    """Dot product operation."""
    def _forward(self,
                 input_1: Tensor,
                 input_2: Tensor
                 ) -> Tensor:
        """Forward pass of the operation."""
        # Check that the second input is a tensor.
        if not isinstance(input_2, Tensor):
            raise ValueError("Dot operation only takes two tensors as input.")

        require_grad = input_1.requires_grad or input_2.requires_grad
        output = Tensor(input_1.data.dot(input_2.data),
                        requires_grad=require_grad,
                        grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gradient of the operation."""
        input_1 = self.inputs[0]
        input_2 = self.inputs[1]

        grad_1 = grad.dot(input_2.data.T)
        grad_2 = input_1.data.T.dot(grad)

        return grad_1, grad_2
