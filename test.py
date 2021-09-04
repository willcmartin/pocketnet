from pocketnet.tensor import Tensor, Linear, MSELoss, SGD #, CrossEntropyLoss
import numpy as np
import sys
import cv2

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
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

X_train = (X_train/255)
Y_train_new = np.zeros((Y_train.shape[0], 10))
for n, y in enumerate(Y_train):
    Y_train_new[n,y] = 1
Y_train = Y_train_new

X_test = (X_test/255)
Y_test_new = np.zeros((Y_test.shape[0], 10))
for n, y in enumerate(Y_test):
    Y_test_new[n,y] = 1
Y_test = Y_test_new

input_size = 784
num_classes = 10
learning_rate = 0.001
BS = 100
num_epochs = 30

model = Linear(input_size, num_classes)

criterion = MSELoss()
optimizer = SGD([model.weight, model.bias], lr=learning_rate)

for epoch in range(num_epochs):
    for i in range(600):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X = Tensor(X_train[samp].reshape((-1, 28*28)))
        Y = Tensor(Y_train[samp])

        outputs = model(X)
        loss = criterion(outputs, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Loss: {:.4f}'
           .format(epoch+1, num_epochs, loss.data))


correct = 0
total = 0
X_test = X_test.reshape((-1, 28*28))

for images, labels in zip(X_test, Y_test):
    images = images.reshape(-1, input_size)
    images = Tensor(images)
    outputs = model(images)

    predicted = np.argmax(outputs.data, 1)
    total += 1

    label = np.argmax(labels, 0)
    if predicted == label:
        correct += 1

print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


test_img = cv2.imread('/Users/willmartin/Desktop/7.jpeg')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
test_img = (255-test_img)
for i in range(test_img.shape[0]):
    for j in range(test_img.shape[1]):
        if test_img[i,j] < 100:
            test_img[i,j] = 0
test_img = (test_img/255)

cv2.imshow("img", test_img)
cv2.waitKey(0)
test_img = cv2.resize(test_img, (28, 28))
test_img = test_img.reshape(-1, input_size)
img = Tensor(test_img)
out = model(img)
print(out.data)
print(np.argmax(out.data, 1))

# input_size = 784
# hidden_size = 500
# output_size = 10
# num_epochs = 100
# lr = 0.001
#
# # x = Tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
# #                     [9.779], [6.182], [7.59], [2.167], [7.042],
# #                     [10.791], [5.313], [7.997], [3.1]])
# #
# # y = Tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
# #                     [3.366], [2.596], [2.53], [1.221], [2.827],
# #                     [3.465], [1.65], [2.904], [1.3]])
#
# class NeuralNet():
#     def __init__(self, input_size, hidden_size, output_size):
#         # super(NeuralNet, self).__init__()
#         self.fc1 = Linear(input_size, hidden_size)
#         self.fc2 = Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.fc2(out)
#         return out
#
#
# # linear = Linear(1, 1)
# model = NeuralNet(input_size, hidden_size, output_size)
#
# # model.fc1.weight.data = np.asarray([[0.5550]])
# # model.fc1.bias.data = np.asarray([0.0455])
# # model.fc2.weight.data = np.asarray([[0.6703]])
# # model.fc2.bias.data = np.asarray([0.8395])
#
# # print ('w1: ', model.fc1.weight.data)
# # print ('b1: ', model.fc1.bias.data)
# # print ('w2: ', model.fc2.weight.data)
# # print ('b2: ', model.fc2.bias.data)
#
#
# # criterion = MSELoss()
# criterion = CrossEntropyLoss()
# optimizer = SGD([model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias], lr=lr)
# # print ('w: ', linear.weight.data)
# # print ('b: ', linear.bias.data)
#
# for epoch in range(num_epochs):
#
#     # Forward pass
#     pred = model.forward(X_train)
#     # pred.data = np.argmax(pred.data, axis=1)
#     # print(pred.data.shape, Y_train.data.shape)
#     loss = criterion(pred, Y_train)
#
#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch+1) % 5 == 0:
#         print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.data))
#
#
#
#
#
# # pred = linear(x)
# # print("pred linear: ", pred.data)
# #
# #
# # loss = criterion(pred, y)
# # print("loss: ", loss.data)
# #
# # loss.backward()
# # print ('dL/dw: ', linear.weight.grad.data)
# # print ('dL/db: ', linear.bias.grad.data)
# #
# #
# # optimizer.step()
# #
# # pred = linear(x)
# # print("step 1- pred linear: ", pred.data)
# #
# # loss = criterion(pred, y)
# # print("step 1- loss: ", loss.data)
# #
# # optimizer.zero_grad()
