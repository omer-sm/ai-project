from DL7 import *
import cupy as cp
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(1)
cp.random.seed(1)
mnist = fetch_openml('mfeat-pixel',as_frame=False, parser="liac-arff")
digits = mnist["data"]

m = 1800
x_train, x_test = digits[:m], digits[m:]
noise_factor = 0.6
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip the images to the valid pixel range
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Normalize the pixel values
'''x_train_noisy = x_train_noisy / 255.
x_test_noisy = x_test_noisy / 255.
x_train = x_train / 255.
x_test = x_test / 255.'''

x_train = cp.array(x_train.T)
x_train_noisy = cp.array(x_train_noisy.T)
x_test_noisy = cp.array(x_test_noisy.T)

model = DLModel("", use_cuda=True)
model.add(DLLayer("1", 224, (240,), "trim_sigmoid", "Xavier", 0.01, "adam"))
model.add(DLLayer("2", 156, (224,), "leaky_relu", "He", 0.01, "adam"))
model.add(DLLayer("3", 224, (156,), "leaky_relu", "He", 0.01, "adam"))
model.add(DLLayer("4", 240, (224,), "trim_sigmoid", "Xavier", 0.01, "adam"))
model.compile("cross_entropy")

costs = model.train(x_train_noisy, x_train, 1000, 300)
denoised_train = cp.asnumpy(model.forward_propagation(x_train_noisy).T)
denoised_test = cp.asnumpy(model.forward_propagation(x_test_noisy).T)
x_train = cp.asnumpy(x_train.T)
x_train_noisy = cp.asnumpy(x_train_noisy.T)
x_test_noisy = cp.asnumpy(x_test_noisy.T)
for i in range(10):
    ind = np.random.randint(0,200)
    print(ind)
    plt.imshow(x_train[ind].reshape(16,15), cmap = matplotlib.cm.binary)
    plt.show()
    plt.imshow(x_train_noisy[ind].reshape(16,15), cmap = matplotlib.cm.binary)
    plt.show()
    plt.imshow(denoised_train[ind].reshape(16,15), cmap = matplotlib.cm.binary)
    plt.show()
    plt.imshow(x_test[ind].reshape(16,15), cmap = matplotlib.cm.binary)
    plt.show()
    plt.imshow(x_test_noisy[ind].reshape(16,15), cmap = matplotlib.cm.binary)
    plt.show()
    plt.imshow(denoised_test[ind].reshape(16,15), cmap = matplotlib.cm.binary)
    plt.show()
#plt.imshow(x_train_noisy[12:12+1].reshape(28,28), cmap = matplotlib.cm.binary)
