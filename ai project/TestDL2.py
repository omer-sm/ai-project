import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from unit10 import c1w4_utils as u10
from DL2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)
train_x_orig, train_y, test_x_orig, test_y, classes = u10.load_datasetC1W4()

m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
num_px = train_x_orig.shape[1]

train_x_flatten = (train_x_orig.reshape(train_x_orig.shape[0], -1)).T
test_x_flatten = (test_x_orig.reshape(test_x_orig.shape[0], -1)).T
train_x = train_x_flatten/255 - 0.5
test_x = test_x_flatten/255 - 0.5

model = DLModel("model")
model.add(DLLayer("l", 30, (train_x.shape[0], ), W_initialization="Xavier", alpha=0.0075))
model.add(DLLayer("l", 15, (30, ), W_initialization="Xavier", alpha=0.0075))
model.add(DLLayer("l", 10, (15, ), W_initialization="Xavier", alpha=0.0075))
model.add(DLLayer("l", 10, (10, ), W_initialization="Xavier", alpha=0.0075))
model.add(DLLayer("l", 5, (10, ), W_initialization="Xavier", alpha=0.0075))
model.add(DLLayer("l", 1, (5, ), W_initialization="Xavier", alpha=0.0075, activation="sigmoid"))
model.compile("cross_entropy")
costs = model.train(train_x, train_y,2500)

plt.plot(np.squeeze(costs))

plt.ylabel('cost')

plt.xlabel('iterations (per 25s)')

plt.title("Learning rate =" + str(0.007))

plt.show()

print("train accuracy:", np.mean(model.predict(train_x) == train_y))

print("test accuracy:", np.mean(model.predict(test_x) == test_y))
img = Image.open(r"C:\Users\omerg\Favorites\Downloads\Belle-Dauphine-1024x1024.jpg")
img = img.convert('RGB')
img = img.resize((64, 64), Image.LANCZOS)
my_image = np.array(img).reshape(1, -1).T
my_image = my_image/255. - 0.5
p = model.predict(my_image)
print ("L-layer model predicts a \"" + classes[int(p),].decode("utf-8") + "\"picture.")