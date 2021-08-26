# from escher.ops import *
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

    def sum(self):
        return Sum.forward(self)

    def mean(self):
        return Mean.forward(self)

    def power(self, other):
        return Power.forward(self, other)

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


    # def backward(self):
    #     if self.grad is None:
    #         self.grad = Tensor(np.ones(self.data.shape))
    #     if self.op is not None:
    #         children_grads = self.op.backward(self, *self.children)
    #         for node, grad in zip(self.children, children_grads):
    #             node.grad = grad
    #     for node in self.children:
    #         node.backward()

class Op:
    @classmethod
    def forward(cls, *inputs):
        parent = Tensor(cls._f(*inputs), list(inputs))
        parent.op = cls
        return parent

    @classmethod
    def backward(cls, parent, children):
        return cls._b(parent, *children)


class Matmul(Op):
    @staticmethod
    def _f(a, b):
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
        print(np.power(a.data, b - 1))
        return [b * np.power(a.data, b - 1) * parent.grad.data]
