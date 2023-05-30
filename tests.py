from linear import *
from activations import *

np.random.seed(0)


if __name__ == "__main__":
    input = Linear(2, 2, Sigmoid())
    middle = Linear(2, 2, Sigmoid())
    out = Linear(2, 2, Sigmoid())

    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    print(inputs.shape)
    outputs = np.array([[1, 0],[0, 1],[0, 1],[1, 0]])

    for epoch in range(20000):
        x = input(inputs)
        x = middle(x)
        r = out(x)

        e = (outputs - r)
        e = out.backward(e, 0.1)
        e = middle.backward(e, 0.1)
        e = input.backward(e, 0.1)

    x = input(inputs)
    x = middle(x)
    print(out(x))
