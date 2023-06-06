import numpy as np

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value, grad):
        self.value = value
        self.grad = grad

class Layer:
    def __init__(self) -> None:
        self._weights = None
        self._bias = None
        self.grad_w = None
        self.grad_b = None

    def params(self):
        return {"W": Param(self._weights, self.grad_w), "B": Param(self._bias, self.grad_b)}