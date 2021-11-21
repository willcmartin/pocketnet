from pocketnet.tensor import Tensor
from pocketnet.nn import Linear, Module, MSELoss
from pocketnet.optim import SGD
import numpy as np

# source (in pytorch):
# https://github.com/yunjey/pytorch-tutorial

def autograd_1():
    x = Tensor([[1.0]])
    w = Tensor([[2.0]])
    b = Tensor([[3.0]])

    y = w.matmul(x).add(b)

    y.backward()

    print("x grad: ", x.grad.data)    # x.grad = 2
    print("w grad: ", w.grad.data)    # w.grad = 1
    print("b grad: ", b.grad.data)    # b.grad = 1

def autograd_2():
    x = Tensor([[ 0.6328, -0.2425],
                [ 0.0692,  0.5727],
                [ 0.0692,  0.5727]])
    y = Tensor([[-0.8524, -0.1387],
                [-0.9748, -0.8326],
                [ 0.0692,  0.5727]])

    linear = Linear(2, 2)
    print ('w: ', linear.weight.data)
    print ('b: ', linear.bias.data)

    linear.weight.data = np.asarray([[-0.0536, -0.2003], [ 0.3518,  0.3669]])
    linear.bias.data = np.asarray([[0.3817, 0.6240]])

    learning_rate = 0.001
    criterion = MSELoss()
    optimizer = SGD([linear.weight, linear.bias], lr=learning_rate)

    pred = linear(x)
    print ('pred: ', pred.data)

    loss = criterion(pred, y)
    print('loss: ', loss.data)

    loss.backward()

    print ('dL/dw: ', linear.weight.grad.data)
    print ('dL/db: ', linear.bias.grad.data)

    optimizer.step()

    pred = linear(x)
    print("pred: ", pred.data)
    loss = criterion(pred, y)
    print('loss after 1 step optimization: ', loss.data)

def linear_regression():
    # Hyper-parameters
    input_size = 1
    output_size = 1
    num_epochs = 60
    learning_rate = 0.001

    # Dataset
    x_train = Tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]])

    y_train = Tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]])

    # Linear regression model
    model = Linear(input_size, output_size)

    # Setting weight and bias for testing
    model.weight.data = np.asarray([[-0.2102]])
    model.bias.data = np.asarray([[0.9135]])

    # Loss and optimizer
    criterion = MSELoss()
    optimizer = SGD([model.weight, model.bias], lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):

        inputs = x_train
        targets = y_train

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.data))


if __name__ == "__main__":
    # autograd_1()
    # autograd_2()
    linear_regression()
