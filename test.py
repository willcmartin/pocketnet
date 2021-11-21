from pocketnet.tensor import Tensor
from pocketnet.nn import Linear, Module, MSELoss
from pocketnet.optim import SGD
import numpy as np
import sys

def basic_linear():
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

basic_linear()


# x = Tensor([[3.44, 6.88, 55.8, 55.8], [3.4, 55.8,2.4, 5.0]])
# b = Tensor([3.44, 6.88, 55.8, 55.8])
#
# # y = x.add(b.max()).sum()
# # y = x.matmul(b).sum()
# #
# y = x.add(b).sum()
#
# print(y.data)
#
# y.backward()
#
# print(x.grad.data)
# print(b.grad.data)




# from https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb
# def fetch(url):
#   import requests, gzip, os, hashlib, numpy
#   fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
#   if os.path.isfile(fp):
#     with open(fp, "rb") as f:
#       dat = f.read()
#   else:
#     with open(fp, "wb") as f:
#       dat = requests.get(url).content
#       f.write(dat)
#   return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
# X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
# Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
# X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
# Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
#
# X_train = (X_train/255)
# Y_train_new = np.zeros((Y_train.shape[0], 10))
# for n, y in enumerate(Y_train):
#     Y_train_new[n,y] = 1
# Y_train = Y_train_new
#
# X_test = (X_test/255)
# Y_test_new = np.zeros((Y_test.shape[0], 10))
# for n, y in enumerate(Y_test):
#     Y_test_new[n,y] = 1
# Y_test = Y_test_new
#
# input_size = 784
# num_classes = 10
# hidden_size = 500
# learning_rate = 0.001
# BS = 100
# num_epochs = 5
#
# class NeuralNet(Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super().__init__()
#         self.fc1 = Linear(input_size, num_classes)
#         # self.relu = ReLU()
#         # self.fc2 = Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         out = self.fc1(x).logsoftmax()
#         # out = self.relu(out)
#         # out = self.fc2(out)
#         return out
#
#
# model = NeuralNet(input_size, hidden_size, num_classes)
#
# criterion = MSELoss()
# optimizer = SGD(model.parameters(), lr=learning_rate)
#
#
# for epoch in range(num_epochs):
#     for i in range(600):
#         samp = np.random.randint(0, X_train.shape[0], size=(BS))
#         X = Tensor(X_train[samp].reshape((-1, 28*28)))
#         Y = Tensor(Y_train[samp])
#
#         outputs = model.forward(X)
#         loss = criterion(outputs, Y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print ('Epoch [{}/{}], Loss: {:.4f}'
#            .format(epoch+1, num_epochs, loss.data))
#
#
# correct = 0
# total = 0
# X_test = X_test.reshape((-1, 28*28))
#
# for images, labels in zip(X_test, Y_test):
#     images = images.reshape(-1, input_size)
#     images = Tensor(images)
#     outputs = model.forward(images)
#
#     predicted = np.argmax(outputs.data, 1)
#     total += 1
#
#     label = np.argmax(labels, 0)
#     if predicted == label:
#         correct += 1
#
# print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
