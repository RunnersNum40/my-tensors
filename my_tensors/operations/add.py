import numpy as np
from operation import Operation, Tensor, Scalar
from typing import Tuple, Union


class Add(Operation):
    """Add operation."""
    def _forward(self,
                 input_1: Tensor,
                 input_2: Union[Tensor, Scalar]
                 ) -> Tensor:
        """Forward pass of the operation."""
        # If the second input is a tensor, add the two tensors element-wise.
        if isinstance(input_2, Tensor):
            require_grad = input_1.requires_grad or input_2.requires_grad
            output = Tensor(input_1.data + input_2.data,
                            requires_grad=require_grad,
                            grad_fn=self)
        # If the second input is a scalar, add the scalar to the tensor.
        else:
            output = Tensor(input_1.data + input_2,
                            requires_grad=input_1.requires_grad,
                            grad_fn=self)
        return output

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gradient of the operation."""
        input_1 = self.inputs[0]
        input_2 = self.inputs[1]

        grad_1 = np.ones_like(input_1.data)
        if isinstance(input_2, Tensor):
            grad_2 = np.ones_like(input_2.data)
        else:
            grad_2 = None

        return grad_1, grad_2


if __name__ == "__main__":
    # Create a tensor.
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), requires_grad=True)
    # Create a scalar.
    y = 2
    # Add the tensor and the scalar.
    z = Add()(x, y)
    # Print the output.
    print(z)
    # Print the gradient.
    z.backward()
    print(x.grad)
