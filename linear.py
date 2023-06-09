import numpy as np
from activations import *


class Linear:
    def __init__(self, inputs, outputs, activaton: ActivationFunc) -> None:
        self.weights = np.random.uniform(size=(inputs, outputs), low = -1)
        self.bias = np.random.uniform(size=1, low = -1)[0]
        self.act = activaton
        self.vals = np.zeros(shape=(outputs))
        self.inputs = np.zeros(shape=(inputs))
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        self.inputs = x
        self.vals = self.act(np.dot(x, self.weights)+ self.bias)
        return self.vals
    
    def backward(self, err, lr):
        d_predicted = err * self.act.derivative(self.vals)
        res = np.dot(d_predicted, self.weights.T)
        self.weights += np.dot( self.inputs.T, d_predicted) * lr
        self.bias += np.sum(d_predicted) * lr
        return res

