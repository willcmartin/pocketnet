# from escher.ops import *
import numpy as np

class Tensor:
    def __init__(self, data, children=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.children = children
        self.grad = None
        self.op = None

    # def __mul__(self, other):
    #     op = Multiply
    #     output = Tensor(op.forward(self, other))
    #     output.op = op
    #     return output

    def matmul(self, other):
        return Matmul.forward(self, other)
        # op = Matmul
        # output = Tensor(*op.forward(self, other))
        # output.op = op
        # return output

    # def __add__(self, other):
    #     op = Add
    #     output = Tensor(op.forward(self, other))
    #     output.op = op
    #     return output
    #
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
        return Tensor(cls._f(*inputs), list(inputs))

class Matmul(Op):
    @staticmethod
    def _f(a, b):
        return np.matmul(a.data, b.data)

    # def _b(parent, a, b):
    #     return [b * parent.grad, a * parent.grad]
