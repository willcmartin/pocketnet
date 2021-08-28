from pocketnet.tensor import Tensor, Linear, MSELoss, SGD
import numpy as np

x = Tensor([[ 0.6328, -0.2425],
        [ 0.0692,  0.5727]])

y = Tensor([[-0.8524, -0.1387],
        [-0.9748, -0.8326]])

linear = Linear(2, 2)
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
