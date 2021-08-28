from pocketnet.tensor import Tensor, Linear, MSELoss, SGD
import numpy as np

num_epochs = 60
lr = 0.001

x = Tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]])

y = Tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]])

linear = Linear(1, 1)
criterion = MSELoss()
optimizer = SGD([linear.weight, linear.bias], lr=lr)
# print ('w: ', linear.weight.data)
# print ('b: ', linear.bias.data)

for epoch in range(num_epochs):

    # Forward pass
    pred = linear(x)
    loss = criterion(pred, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.data))





# pred = linear(x)
# print("pred linear: ", pred.data)
#
#
# loss = criterion(pred, y)
# print("loss: ", loss.data)
#
# loss.backward()
# print ('dL/dw: ', linear.weight.grad.data)
# print ('dL/db: ', linear.bias.grad.data)
#
#
# optimizer.step()
#
# pred = linear(x)
# print("step 1- pred linear: ", pred.data)
#
# loss = criterion(pred, y)
# print("step 1- loss: ", loss.data)
#
# optimizer.zero_grad()
