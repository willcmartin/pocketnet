# from pocketnet.ops import *
import numpy as np
import math

class Tensor:
    def __init__(self, data, children=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.children = children
        self.grad = None
        self.op = None

    def matmul(self, other):
        return Matmul.forward(self, other)

    # def multiply(self, other):
    #     return Multiply.forward(self, other)
        # out = self
        # out.data = np.multiply(out.data, other)
        # return out

    def abs(self):
        out = self
        out.data = np.abs(out.data)
        return out

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
        return Transpose.forward(self)

    # def log(self):
    #     return Log.forward(self)
    #
    # def exp(self):
    #     return Exp.forward(self)

    def relu(self):
        return ReluOp.forward(self)

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

# TODO: learn about classmethod and staticmethod

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

def unbroadcast(in_grad, out_shape):
    # TODO: make more flexible
    if in_grad.shape != out_shape:
        out_grad = np.sum(in_grad, axis=0)
    else:
        out_grad = in_grad
    return out_grad

class Matmul(Op):
    @staticmethod
    def _f(a, b):
        # TODO: expand dim
        return np.matmul(a.data, b.data)

    @staticmethod
    def _b(parent, a, b):
        return [parent.grad.data @ b.data.T , a.data.T @ parent.grad.data]

# class Multiply(Op):
#     # TODO: check all of this
#     @staticmethod
#     def _f(a, b):
#         # TODO: expand dim
#         return np.multiply(a.data, b.data)
#
#     @staticmethod
#     def _b(parent, a, b):
#         return [parent.grad.data * b.data , parent.grad.data * a.data]

class Add(Op):
    @staticmethod
    def _f(a, b):
        return np.add(a.data, b.data)

    @staticmethod
    def _b(parent, a, b):
        return [unbroadcast(parent.grad.data, a.data.shape),
                unbroadcast(parent.grad.data, b.data.shape)]

class Subtract(Op):
    @staticmethod
    def _f(a, b):
        return np.subtract(a.data, b.data)

    @staticmethod
    def _b(parent, a, b):
        return [unbroadcast(parent.grad.data, a.data.shape),
                unbroadcast(-parent.grad.data, b.data.shape)]

class Sum(Op):
    @staticmethod
    def _f(a):
        return np.sum(a.data)

    @staticmethod
    def _b(parent, a):
        return [np.ones(a.data.shape)]

class Mean(Op):
    @staticmethod
    def _f(parent):
        return np.mean(parent.data)

    @staticmethod
    def _b(parent, a):
        return [np.ones(a.data.shape)*(1/a.data.size)]

class Power(Op):
    @staticmethod
    def _f(a, b):
        return np.power(a.data, b)

    @staticmethod
    def _b(parent, a, b):
        return [b * np.power(a.data, b - 1) * parent.grad.data]

class Transpose(Op):
    @staticmethod
    def _f(a):
        return np.transpose(a.data)

    @staticmethod
    def _b(parent, a):
        return [np.transpose(parent.grad.data)]

class ReluOp(Op):
    @staticmethod
    def _f(a):
        return np.maximum(0, a.data)

    @staticmethod
    def _b(parent, a):
        return [parent.grad.data * (a.data >= 0)]


# class Log(Op):
#     @staticmethod
#     def _f(a):
#         return np.log(a.data)
#
#     # TODO: Check math!
#     def _b(parent, a):
#         return [parent.grad.data / a.data]

# class Exp(Op):
#     @staticmethod
#     def _f(a):
#         return np.exp(a.data)
#
#     # TODO: Check math!
#     def _b(parent, a):
#         return [a.data * parent.grad.data]



################################################################################

class Linear:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        # initialize better: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
        stdv = 1. / math.sqrt(in_dim*out_dim)
        self.weight = Tensor(np.random.uniform(-stdv, stdv, (out_dim, in_dim)))
        self.bias = Tensor(np.random.uniform(-stdv, stdv, (out_dim)))

    def __call__(self, x):
        return x.matmul(self.weight.transpose()).add(self.bias)

class ReLU:
    def __init__(self):
        pass

    def __call__(self, x):
        return x.relu()


################################################################################

class MSELoss:
    def __init__(self):
        pass

    def __call__(self, pred, true):
        return (true.subtract(pred).power(2).mean())

# class CrossEntropyLoss:
#     def __init__(self):
#         pass
#     def __call__(self, pred, true):
#         # print(pred.data)
#         # print((true.matmul(pred.log()).sum()).data)
#         # return (pred.multiply(-1).transpose().matmul(true).mean())
#         # print(pred.abs().log().data)
#         # print(pred.data)
#         return(true.multiply(pred.abs().log()).mean())

################################################################################

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
