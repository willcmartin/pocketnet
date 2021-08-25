from escher.tensor import Tensor
import numpy as np

x = Tensor(np.asarray([[15]]))
print(f'x: {x.data}')

y = x.matmul(x)
print(f'y: {y.data}')
#
# y.backward()
#
# print(x.grad.data)
