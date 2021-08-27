# from pocketnet.ops import *
import numpy as np

class Tensor:
    def __init__(self, data, children=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.children = children
        self.grad = None
        self.op = None

    def matmul(self, other):
        return Matmul.forward(self, other)

    def add(self, other):
        return Add.forward(self, other)

    def subtract(self, other):
        return Subtract.forward(self, other)

    def sum(self):
        return Sum.forward(self)

    def mean(self):
        return Mean.forward(self)

    def power(self, other):
        return Power.forward(self, other)

    def transpose(self):
        out = self
        out.data = np.transpose(out.data)
        return out

    def backward(self):
        if self.grad is None:
            self.grad = Tensor(np.ones(self.data.shape))
        if self.op is not None:
            children_grads = self.op.backward(self, self.children)
            for child, grad in zip(self.children, children_grads):
                child.grad = Tensor(grad)
        for child in self.children:
            if isinstance(child, Tensor):
                child.backward()


class Op:
    @classmethod
    def forward(cls, *inputs):
        parent = Tensor(cls._f(*inputs), list(inputs))
        parent.op = cls
        return parent

    @classmethod
    def backward(cls, parent, children):
        return cls._b(parent, *children)

################################################################################

class Matmul(Op):
    @staticmethod
    def _f(a, b):
        # TODO: expand dim
        return np.matmul(a.data, b.data)

    @staticmethod
    def _b(parent, a, b):
        return [parent.grad.data @ b.data.T , a.data.T @ parent.grad.data]

class Add(Op):
    @staticmethod
    def _f(a, b):
        return np.add(a.data, b.data)

    def _b(parent, a, b):
        return [parent.grad.data, parent.grad.data]

class Subtract(Op):
    @staticmethod
    def _f(a, b):
        return np.subtract(a.data, b.data)

    def _b(parent, a, b):
        return [parent.grad.data, -parent.grad.data]

class Sum(Op):
    @staticmethod
    def _f(a):
        return np.sum(a.data)

    def _b(parent, a):
        return [np.ones(a.data.shape)]

class Mean(Op):
    @staticmethod
    def _f(parent):
        return np.mean(parent.data)

    def _b(parent, a):
        return [np.ones(a.data.shape)*(1/a.data.size)]

class Power(Op):
    @staticmethod
    def _f(a, b):
        return np.power(a.data, b)

    def _b(parent, a, b):
        return [b * np.power(a.data, b - 1) * parent.grad.data]

################################################################################

class Linear:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        # TODO: make random
        self.weight = Tensor([[0.3304]])
        self.bias = Tensor([0.4247])

    def __call__(self, x):
        return x.matmul(self.weight.transpose()).add(self.bias)

################################################################################

class MSELoss:
    def __init__(self):
        pass
    def __call__(self, pred, true):
        # TODO: add ops to Tensor
        return (true.subtract(pred).power(2).mean())

################################################################################

class SGD:
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr
    # todo zero grad function
    def step(self):
        for param in self.params:
            print(param.data)
            param.data -= (param.grad.multiply(self.lr))

    def zero_grad(self):
        for param in self.params:
            param.grad.data = 0
