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

    def log(self):
        return Log.forward(self)

    def exp(self):
        return Exp.forward(self)

    def relu(self):
        return ReluOp.forward(self)

    def max(self):
        return Max.forward(self)

    def logsoftmax(self):
        return LogSoftmaxOp.forward(self)

        # x_off = self.subtract(self.max())
        # return x_off.subtract(((x_off.exp()).sum()).log())
        # x_off = self.add(self)
        # return x_off#.subtract((x_off.exp()).sum().log())

    def backward(self):
        if self.grad is None:
            self.grad = Tensor(np.ones(self.data.shape))
        if self.op is not None:
            children_grads = self.op.backward(self, self.children)
            for child, grad in zip(self.children, children_grads):
                child.grad = Tensor(grad) if child.grad is None else Tensor(np.add(child.grad.data, grad)) #child.grad.add(Tensor(grad)) # idk which is correct
        for child in self.children:
            if isinstance(child, Tensor):
                child.backward()

### Ops ###

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
        return np.sum(a.data).reshape(1)

    @staticmethod
    def _b(parent, a):
        return [np.ones(a.data.shape)*parent.grad.data]

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

class LogSoftmaxOp(Op):
    @staticmethod
    def _f(a):
        x_off = a.subtract(a.max())
        return x_off.subtract((x_off.exp()).sum().log())

    @staticmethod
    def _b(parent, a):
        return [parent.grad]

class Log(Op):
    @staticmethod
    def _f(a):
        return np.log(a.data)

    # TODO: Check math!
    @staticmethod
    def _b(parent, a):
        return [parent.grad.data / a.data]

class Exp(Op):
    @staticmethod
    def _f(a):
        return np.exp(a.data)

    # TODO: Check math!
    @staticmethod
    def _b(parent, a):
        return [parent.grad.data * parent.data]

class Max(Op):
    @staticmethod
    def _f(a):
        return  np.max(a.data).reshape(1) # TODO: is reshape needed?

    @staticmethod
    def _b(parent, a):
        out = np.zeros(a.data.shape)
        # if a != 0
        out[np.where(a.data == parent.data)] = 1
        return [out/np.sum(out)]

### Helper functions ###

def unbroadcast(in_grad, out_shape):
    # TODO: make more flexible!
    if in_grad.shape != out_shape:
        sum_axes = []
        for i in range(len(in_grad.shape)):
            try:
                if out_shape[i]==1:
                    sum_axes.append(i)
            except:
                sum_axes.append(i)
        out_grad = np.sum(in_grad, axis=tuple(sum_axes))
    else:
        out_grad = in_grad
    return out_grad
