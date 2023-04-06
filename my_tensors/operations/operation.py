import numpy as np
from typing import Tuple, Union, Any
from my_tensors.tensor import Tensor

Scalar = Union[int, float]
Input = Union[Tensor, Scalar, tuple]


class Operation:
    """Operation base-class.

    Operations are used to represent the computations that are performed on
    tensors. Operations map an arbitrary number of inputs to a single output.

    Forward passes return the output of the operation. Backward passes return
    the gradient of the output of the operation with respect to the inputs to
    the operation.

    A set of operations defines a computational graph. The graph is used to
    calculate the gradient of the output of the graph with respect to the
    inital parameters of the graph.

    Attributes:
        inputs (Tuple[Tensor, ...]): Inputs to the operation.
        output (Tensor): Output of the operation.
    """
    def __init__(self) -> None:
        """Operation constructor.

        Operations are initialized with no inputs or outputs.
        """
        # Initialize the inputs and output to None.
        self.inputs = None
        self.output = None

    def forward(self,
                *inputs: Tuple[Input, ...]
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
        self.inputs = inputs
        self.output = self._forward(*inputs)
        self.output.grad_fn = self

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
