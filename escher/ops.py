import numpy as np
from .tensor import Op

# class Multiply:
#     def forward(a, b):
#         print("a ", a.data, "b ", b.data)
#         return np. multiply(a.data, b.data), [a, b]
#
#     def backward(parent, a, b):
#
#         return [b * parent.grad, a * parent.grad]

class Matmul(Op):
    def forward(a, b):
        print("a ", a.data, "b ", b.data)
        return np.matmul(a.data, b.data), [a, b]

    def backward(parent, a, b):

        return [b * parent.grad, a * parent.grad]

# class Add:
#     def forward(a, b):
#         return np.add(a.data, b.data), [a, b]
#
#     def backward(parent, a, b):
#         return [parent.grad, parent.grad]
#
