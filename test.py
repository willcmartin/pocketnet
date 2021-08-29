from pocketnet.tensor import Tensor, Linear, MSELoss, SGD, CrossEntropyLoss
import numpy as np
import sys

# from https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb
def fetch(url):
  import requests, gzip, os, hashlib, numpy
  fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]

X_train_new = X_train.reshape(-1, 28*28)[:1,:]

X_train = Tensor(X_train_new)

Y_train_new = np.zeros((1,10))

for n, y in enumerate(Y_train[:1]):
    Y_train_new[n,y] = 1

Y_train = Tensor(Y_train_new)


input_size = 784
hidden_size = 500
output_size = 10
num_epochs = 10
lr = 0.001

# x = Tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                     [9.779], [6.182], [7.59], [2.167], [7.042],
#                     [10.791], [5.313], [7.997], [3.1]])
#
# y = Tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
#                     [3.366], [2.596], [2.53], [1.221], [2.827],
#                     [3.465], [1.65], [2.904], [1.3]])

class NeuralNet():
    def __init__(self, input_size, hidden_size, output_size):
        # super(NeuralNet, self).__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


# linear = Linear(1, 1)
model = NeuralNet(input_size, hidden_size, output_size)

# model.fc1.weight.data = np.asarray([[0.5550]])
# model.fc1.bias.data = np.asarray([0.0455])
# model.fc2.weight.data = np.asarray([[0.6703]])
# model.fc2.bias.data = np.asarray([0.8395])

# print ('w1: ', model.fc1.weight.data)
# print ('b1: ', model.fc1.bias.data)
# print ('w2: ', model.fc2.weight.data)
# print ('b2: ', model.fc2.bias.data)


criterion = MSELoss()
# criterion = CrossEntropyLoss()
optimizer = SGD([model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias], lr=lr)
# print ('w: ', linear.weight.data)
# print ('b: ', linear.bias.data)

for epoch in range(num_epochs):

    # Forward pass
    pred = model.forward(X_train)
    # pred.data = np.argmax(pred.data, axis=1)
    # print(pred.data.shape, Y_train.data.shape)
    loss = criterion(pred, Y_train)

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
