from linear import *
from activations import *
import mnist

np.random.seed(0)

class MnistLinear:
    def __init__(self, lr = 0.001) -> None:
        self.layers = [
            Linear(28*28, 16, Sigmoid()),
            Linear(16, 16, Sigmoid()),
            Linear(16, 10, Sigmoid())
        ]
        self.lr = lr
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, expected_output):
        err = expected_output - self.layers[-1].vals
        for layer in reversed(self.layers):
            err = layer.backward(err, self.lr)
        
if __name__ == "__main__":
    #mnist.init()
    x_train, t_train, x_test, t_test = mnist.load()
    model = MnistLinear()

    for epoch in range(50):
        print(f"epoch {epoch}/50")
        for (img, label) in zip(x_train, t_train):
            expected = np.zeros(shape=(10))
            expected[label] = 1

            model.forward(img)
            model.backward(expected)

    successed = 0

    for (img, label) in zip(x_test, t_test):
        if np.argmax(model.forward(img)) == label:
            successed += 1

    print(successed / 10000)
    
