import numpy as np
from operation import Operation, Tensor
from typing import Tuple, Union


class Index(Operation):
    """Index operation."""
    def __init__(self, index: Union[tuple[int, ...], int]):
        """Constructor of the class."""
        super().__init__()
        self.index = index

    def _forward(self, input_1: Tensor) -> Tensor:
        """Forward pass of the operation."""
        # Perform the index operation.
        require_grad = input_1.requires_grad
        output = Tensor(np.array(input_1.data[self.index]),
                        requires_grad=require_grad,
                        grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        """Gradient of the operation."""
        input_1 = self.inputs[0]

        grad_1 = np.zeros_like(input_1.data)
        grad_1[self.index] = grad

        return (grad_1,)
