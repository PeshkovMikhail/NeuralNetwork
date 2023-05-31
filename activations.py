import numpy as np

class ActivationFunc:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x) -> float:
        pass

    def derivative(self, x) -> float:
        pass

class Sigmoid(ActivationFunc):
    def __init__(self) -> None:
        pass

    def __call__(self, x) -> float:
        return 1/(1+np.exp(-x))
    
    def derivative(self, x) -> float:
        return x * (1 - x)
    
class Softmax(ActivationFunc):
    def __init__(self) -> None:
        pass

    def __call__(self, x) -> float:
        return 
    

class ReLu(ActivationFunc):
    def __init__(self) -> None:
        pass

    def __call__(self, x) -> float:
        return np.maximum(0, x)
    
    def derivative(self, x):
        res = x.copy()
        res[res>=0] = 1
        res[res<0] = 0
        return res