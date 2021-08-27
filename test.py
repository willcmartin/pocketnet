from pocketnet.tensor import Tensor, Linear, MSELoss, SGD
import numpy as np

# x = Tensor([[15, 10], [10, 10]])
# w = Tensor([[10, 6],[10, 10]])
# b = Tensor([[900, 900], [900, 900]])
#
# print(f'x: {x.data}')
#
# y = x.matmul(w).subtract(b).power(2).mean()
# print(f'y: {y.data}')
# #
# y.backward()
# #
# print(x.grad.data)
# print(w.grad.data)
# print(b.grad.data)


x = Tensor([[1.1527]])

y = Tensor([[-0.3874]])

linear = Linear(1, 1)
print ('w: ', linear.weight.data)
print ('b: ', linear.bias.data)

pred = linear(x)
print("pred linear: ", pred.data)

criterion = MSELoss()

loss = criterion(pred, y)

print("loss: ", loss.data)

loss.backward()

print ('dL/dw: ', linear.weight.grad.data)
print ('dL/db: ', linear.bias.grad.data)

optimizer = SGD([linear.weight, linear.bias], lr=0.01)

optimizer.step()

pred = linear(x)
print("step 1- pred linear: ", pred.data)

loss = criterion(pred, y)
print("step 1- loss: ", loss.data)
