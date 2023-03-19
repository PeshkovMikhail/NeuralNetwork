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
    

class ReLu(ActivationFunc):
    def __init__(self) -> None:
        pass

    def __call__(self, x) -> float:
        return 0
    
    def derivative(self, x):
        pass