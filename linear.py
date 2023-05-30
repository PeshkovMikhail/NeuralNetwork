import numpy as np
from activations import *


class Linear:
    def __init__(self, inputs, outputs, activaton: ActivationFunc) -> None:
        self.weights = np.random.uniform(low = -1, high=1,size=(inputs, outputs))
        self.bias = np.random.uniform(low = -1, high=1, size=(1, outputs))
        self.act = activaton
        self.vals = None
        self.inputs = None
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        self.inputs = x
        self.vals = np.dot(x, self.weights) + self.bias
        if self.act:
            self.vals = self.act(self.vals)
        return self.vals
    
    def backward(self, err, lr):
        if self.act:
            d_predicted = err * self.act.derivative(self.vals)
        else:
            d_predicted = err
        d_out = np.dot(d_predicted, self.weights.T)
        self.weights -= np.dot( self.inputs.T, d_predicted) * lr
        self.bias -= np.sum(d_predicted, axis = 0, keepdims=True) * lr
        return d_out


