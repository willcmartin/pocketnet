from pocketnet.tensor import Tensor
import numpy as np
import math

class Module:
    def parameters(self):
        parameters = []
        if hasattr(self, '__dict__'):
            for val in self.__dict__.values():
                for subval in val.__dict__.values():
                    if isinstance(subval, Tensor):
                        parameters.append(subval)
        return parameters

class Linear():
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        # initialize better: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
        stdv = 1. / math.sqrt(in_dim*out_dim)
        self.weight = Tensor(np.random.uniform(-stdv, stdv, (out_dim, in_dim)))
        self.bias = Tensor(np.random.uniform(-stdv, stdv, (1, out_dim)))

    def __call__(self, x):
        return x.matmul(self.weight.transpose()).add(self.bias)

class ReLU:
    def __call__(self, x):
        return x.relu()

class LogSoftmax:
    def __init__(self, a, dim=1):
        self.a = a
        self.dim = dim

    def __call__(self):
        x_off = self.a.subtract(self.a.max())
        return (x_off.subtract((x_off.exp()).sum().log()))

class MSELoss:
    def __call__(self, pred, true):
        return (true.subtract(pred).power(2).mean())

class CrossEntropyLoss:
    def __init__(self):
        pass
    def __call__(self, pred, true):
        # print(pred.data)
        # print((true.matmul(pred.log()).sum()).data)
        # return (pred.multiply(-1).transpose().matmul(true).mean())
        # print(pred.abs().log().data)
        # print(pred.data)
        return(true.multiply(pred.abs().log()).mean())
