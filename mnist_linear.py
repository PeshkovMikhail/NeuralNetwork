from linear import *
from activations import *
from loss import CategoricalCrossEntropyLoss
import mnist

np.random.seed(0)

class MnistLinear:
    def __init__(self, lr = 0.001) -> None:
        self.layers = [
            Linear(28*28, 64, Sigmoid()),
            Linear(64, 10, None)
        ]
        self.lr = lr
        self.res = None
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        self.res = x
        return x
    
    def backward(self, expected_output):
        loss = CategoricalCrossEntropyLoss()
        loss_val = loss(self.res, expected_output)
        err = loss.derivatives()
        for layer in reversed(self.layers):
            err = layer.backward(err, self.lr)
        return loss_val
        
if __name__ == "__main__":
    np.random.seed(0)
    #mnist.init()
    x_train, t_train, x_test, t_test = mnist.load()
    model = MnistLinear()
    EPOCHES = 10


    for epoch in range(EPOCHES):
        print(f"epoch {epoch}/{EPOCHES}")
        loss_val = 0
        for i, (img, label) in enumerate(zip(x_train, t_train)):

            res = model.forward(img.reshape((1, 28*28))/255.0)
            loss_val += model.backward(label)
            if i % 10000 == 0 and i > 0:
                print(f"epoch: {epoch}, loss: {loss_val/10000}")
                loss_val = 0
            break
        break

    print(model.layers[0].weights)
    exit(0)


    successed = 0

    results = np.zeros((3, 10))
    for (img, label) in zip(x_train[:1000], t_train[:1000]):
        results[0][label] += 1
        r = model.forward(img.reshape((1, 28*28))/255.0)
        if np.argmax(r, axis = 1) == label:
            results[1][label] +=1
        results[2][np.argmax(r, axis = 1)] += 1

    print(results[1])
    print(results[0])
    print(results[2])
    
