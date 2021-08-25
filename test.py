from escher.tensor import Tensor
import numpy as np

x = Tensor([[15]])
w = Tensor([[10]])
b = Tensor([[900]])

print(f'x: {x.data}')

y = x.matmul(w).add(b)
print(f'y: {y.data}')
#
y.backward()
#
print(x.grad.data)
print(w.grad.data)
print(b.grad.data)
