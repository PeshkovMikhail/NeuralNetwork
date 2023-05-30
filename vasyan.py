from activations import *

s = Sigmoid()
t = s(np.array([1, 0.01]))
print(s.derivative(t))
