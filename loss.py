import numpy as np
def softmax(tensor):
    e = np.exp(tensor)
    return e/np.sum(e, axis= 1, keepdims=True)

class CategoricalCrossEntropyLoss:
    def __init__(self) -> None:
        self.vals = None
        self.label = None

    def __call__(self, predicted, label):
        self.vals = softmax(predicted)
        self.label = label
        return np.mean(-np.log(self.vals[:, label]))

    def derivatives(self):
        res = self.vals.copy()
        res[:, self.label] -= 1
        return res
