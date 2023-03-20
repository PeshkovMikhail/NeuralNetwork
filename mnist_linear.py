from linear import *
from activations import *
import mnist

np.random.seed(0)


class MnistLinear:
    def __init__(self, lr = 0.005) -> None:
        self.layers = [
            Linear(28*28, 32, Sigmoid()),
            Linear(32, 16, Sigmoid()),
            Linear(16, 10, Sigmoid())
        ]
        self.lr = lr
    
    def forward(self, x):
        x = x.reshape((1, 784))
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, expected_output):
        first_err = err = (expected_output - self.layers[-1].vals)
        for layer in reversed(self.layers):
            err = layer.backward(err, self.lr)
        return first_err
        
if __name__ == "__main__":
    #mnist.init()
    x_train, t_train, x_test, t_test = mnist.load()

    index = np.arange(60000)
    np.random.shuffle(index)
    model = MnistLinear()

    for epoch in range(30):
        print(f"epoch {epoch}/100")
        for (i, num) in enumerate(index):
            img = x_train[i]
            label = t_train[i]
            expected = np.zeros(shape=(10))
            expected[label] = 1
            expected = expected.reshape((1, 10))

            model.forward(img)
            er = model.backward(expected)
            if num % 1000 == 0 and epoch == 29:
                print(sum(abs(er[0])))

    successed = [0 for i in range(10)]
    all = [0 for i in range(10)]

    for (img, label) in zip(x_test, t_test):
        if np.argmax(model.forward(img)) == label:
            successed[label] += 1
        all[label] += 1

    for (s, a) in zip(successed, all):
        print(f"{s}/{a}")
    print(sum(successed)/sum(all))
    
