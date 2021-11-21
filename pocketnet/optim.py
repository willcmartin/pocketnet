import numpy as np

class SGD:
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data = param.data - (np.multiply(param.grad.data, self.lr))

    def zero_grad(self):
        for param in self.params:
            param.grad = None
