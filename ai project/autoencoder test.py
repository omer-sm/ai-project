from DL8 import *
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
noise_factor = 0.3
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)


# Normalize the pixel values
x_train_noisy = x_train_noisy / 255.
x_test_noisy = x_test_noisy / 255.
x_train = x_train / 255.
x_test = x_test / 255.

# Clip the images to the valid pixel range
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_train = np.clip(x_train, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

x_train = cp.array(x_train.T)
x_train_noisy = cp.array(x_train_noisy.T)
x_test_noisy = cp.array(x_test_noisy.T)


model = DLModel("", use_cuda=True)
model.add(DLLayer("1", 128, (240,), "leaky_relu", "He", 0.002, "adam"))
model.add(DLLayer("1", 64, (128,), "trim_tanh", "Xavier", 0.002, "adam"))
model.add(DLLayer("2", 16, (64,), "vae_bottleneck", "He", 0.002, "adam", samples_per_dim=6))
model.add(DLLayer("3", 128, (48,), "leaky_relu", "He", 0.002, "adam"))
model.add(DLLayer("4", 240, (128,), "trim_sigmoid", "Xavier", 0.002, "adam"))
model.compile("cross_entropy_KLD", recon_loss_weight=0.7)
costs = model.train(x_train_noisy, x_train, 600, 600)
encoded = cp.array(digits).T
for l in range(1,4):
    encoded = model.layers[l].forward_propagation(encoded, False)
encoded = cp.asnumpy(encoded)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(encoded[0] / np.max(np.abs(encoded[0])), encoded[1] / np.max(np.abs(encoded[1])), encoded[2] / np.max(np.abs(encoded[2])), c=np.array(mnist["target"]).astype(int))
plt.show()
gen = cp.random.normal(0.5, 0.5, (48,4))
for l in range(4,6):
    gen = model.layers[l].forward_propagation(gen, False)
gen = cp.asnumpy(gen).T
for i in range(2):
    plt.imshow(gen[i].reshape(16,15), cmap = matplotlib.cm.binary)
    plt.show()
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
