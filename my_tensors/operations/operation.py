import numpy as np
from ..tensor import Tensor
from typing import Tuple, Union, Any


Scalar = Union[int, float]


class Operation:
    """Operation base-class.

    Operations are used to represent the computations that are performed on
    tensors in a neural network. Operations are used to represent the forward
    and backward passes of a neural network.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def __init__(self) -> None:
        """Constructor."""
        # Initialize the inputs and output to None.
        self.inputs = None
        self.output = None

    def forward(self,
                *inputs: Tuple[Union[Tensor, Scalar, tuple], ...]
                ) -> Tensor:
        """Forward pass of the operation.

        Args:
            inputs (Tuple[Tensor, ...]): Inputs to the operation.

        Returns:
            Tensor: Output of the operation.

        Raises:
            TypeError: If any of the inputs are numpy arrays.
            TypeError: If the first input is not a tensor.
        """
        if any(isinstance(input_, np.ndarray) for input_ in inputs):
            raise TypeError("Convert numpy arrays to tensors before \
                            passing them to operations.")

        if not isinstance(inputs[0], Tensor):
            raise TypeError("Inputs to operations must be tensors.")

        self.inputs = inputs
        self.output = self._forward(*inputs)

        return self.output

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the operation.

        Args:
            grad (Tensor): Gradient of the loss with respect to the output of
                the operation.

        Returns:
            Tuple[Tensor, ...]: Gradient of the loss with respect to the inputs
                to the operation.

        Raises:
            TypeError: If the gradient is not a numpy array.
        """
        if not isinstance(grad, np.ndarray):
            raise TypeError("Gradients must be numpy arrays.")

        return self._backward(grad)

    def _forward(self,
                 *inputs: Tuple[Union[Tensor, Scalar, tuple], ...]
                 ) -> Tensor:
        """Forward pass of the operation."""
        raise NotImplementedError

    def _backward(self, grad: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Gradient of the operation."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call the operation."""
        return self.forward(*args, **kwds)

    def __repr__(self) -> str:
        """Representation."""
        return f"{self.__class__.__name__}()"
