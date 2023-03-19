from linear import *
from activations import *

np.random.seed(0)


if __name__ == "__main__":
    input = Linear(2, 2, Sigmoid())
    out = Linear(2, 1, Sigmoid())

    inputs = [
        np.array([[0, 0]]),
        np.array([[1, 0]]),
        np.array([[0, 1]]),
        np.array([[1, 1]])
    ]

    outputs = [0, 1, 1, 0]

    for epoch in range(10000):
        for (i, o) in zip(inputs, outputs):
            x = input(i)
            r = out(x)

            e = o - r
            e = out.backward(x, e, 0.4)
            e = input.backward(i, e, 0.4)
            if epoch % 1000 == 0:
                print(e, i)
                print("------")

    for i in inputs:
        x = input(i)
        print(out(x))
