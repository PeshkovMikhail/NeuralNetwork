import numpy as np
from activations import *
from layer import *

class Linear(Layer):
    def __init__(self, inputs, outputs, activaton: ActivationFunc) -> None:
        super().__init__()
        sqrt_k = np.sqrt(1/inputs)
        self._weights = np.random.uniform(low = -sqrt_k, high=sqrt_k,size=(inputs, outputs))
        self.grad_w = np.zeros_like(self._weights)
        self._bias = np.random.uniform(low = -sqrt_k, high=sqrt_k, size=(1, outputs))
        self.grad_b = np.zeros_like(self._bias)
        self.act = activaton
        self.vals = None
        self.inputs = None
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        self.inputs = x
        self.vals = np.dot(x, self._weights) + self._bias
        if self.act:
            self.vals = self.act(self.vals)
        return self.vals
    
    def backward(self, err, lr):
        if self.act:
            d_predicted = err * self.act.derivative(self.vals)
        else:
            d_predicted = err
        d_out = np.dot(d_predicted, self._weights.T)
        self.grad_w = np.dot( self.inputs.T, d_predicted)
        self._weights -= self.grad_w * lr
        self.grad_b = np.sum(d_predicted, axis = 0, keepdims=True)
        self._bias -= self.grad_b * lr
        return d_out


