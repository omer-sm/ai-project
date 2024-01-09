from DL8 import *
import cupy as cp
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib
import os, os.path
from PIL import Image

np.random.seed(1)
cp.random.seed(1)

r'''imgs = []
path = r"C:\Users\omerg\Favorites\Downloads\cats"
for i in range(6):
    for f in os.listdir(path)[i*3000:(i+1)*3000]:
        ext = os.path.splitext(f)[1]
        if ext.lower() != ".jpg":
            continue
        img = np.array(Image.open(os.path.join(path,f)).convert("L")).flatten()
        imgs.append(img)
imgs = cp.array(imgs)[:15000].T
#imgs = imgs / 255.

model = DLModel("", use_cuda=True)
model.add(DLLayer("1", 128, (4096,), "leaky_relu", "He", 0.002, "adam"))
model.add(DLLayer("2", 80, (128,), "vae_bottleneck", "Xavier", 0.001, "adam", samples_per_dim=2))
model.add(DLLayer("3", 64, (80,), "leaky_relu", "He", 0.002, "adam"))
model.add(DLLayer("4", 4096, (64,), "relu", "He", 0.002, "adam"))
model.compile("squared_means_KLD", recon_loss_weight=0.8)
costs = model.train(imgs, imgs, 100, 2500)
processed = (cp.asnumpy(model.forward_propagation(imgs)).T).astype(int)
imgs = cp.asnumpy(imgs.T ).astype(int)

for i in range(20):
    plt.imshow(imgs[i*np.random.randint(1,100)].reshape(64,64))
    plt.show()
    plt.imshow(processed[i*np.random.randint(1,100)].reshape(64,64))
    plt.show()'''

mnist = fetch_openml('mnist_784',as_frame=False, parser="liac-arff")
digits = mnist["data"]
m = 60000
x_train, x_test = digits[:m], digits[m:]
noise_factor = 0.
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
model.add(DLLayer("1", 128, (784,), "trim_tanh", "He", 0.002, "adam"))
model.add(DLLayer("1", 64, (128,), "trim_tanh", "Xavier", 0.002, "adam"))
model.add(DLLayer("2", 32, (64,), "vae_bottleneck", "He", 0.002, "adam", samples_per_dim=6))
model.add(DLLayer("3", 128, (96,), "leaky_relu", "He", 0.002, "adam"))
model.add(DLLayer("4", 784, (128,), "trim_sigmoid", "Xavier", 0.002, "adam"))
model.compile("cross_entropy_KLD", recon_loss_weight=0.8)
costs = model.train(x_train_noisy, x_train, 250, 12000)
costs = cp.asnumpy(costs)
plt.plot(costs)
plt.show()
encoded = cp.array(digits[::100]).T
for l in range(1,4):
    encoded = model.layers[l].forward_propagation(encoded, False)
encoded = cp.asnumpy(encoded)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(encoded[0] / np.max(np.abs(encoded[0])), encoded[1] / np.max(np.abs(encoded[1])), encoded[2] / np.max(np.abs(encoded[2])), c=np.array(mnist["target"])[::100].astype(int))
plt.show()
gen = cp.random.normal(0.5, 0.5, (96,4))
for l in range(4,6):
    gen = model.layers[l].forward_propagation(gen, False)
gen = cp.asnumpy(gen).T
for i in range(4):
    plt.imshow(gen[i].reshape(28,28), cmap = matplotlib.cm.binary)
    plt.show()
denoised_train = cp.asnumpy(model.forward_propagation(x_train_noisy).T)
denoised_test = cp.asnumpy(model.forward_propagation(x_test_noisy).T)
x_train = cp.asnumpy(x_train.T)
x_train_noisy = cp.asnumpy(x_train_noisy.T)
x_test_noisy = cp.asnumpy(x_test_noisy.T)
for i in range(10):
    ind = np.random.randint(0,200)
    print(ind)
    plt.imshow(x_train[ind].reshape(28,28), cmap = matplotlib.cm.binary)
    plt.show()
    #plt.imshow(x_train_noisy[ind].reshape(16,15), cmap = matplotlib.cm.binary)
    #plt.show()
    plt.imshow(denoised_train[ind].reshape(28,28), cmap = matplotlib.cm.binary)
    plt.show()
    plt.imshow(x_test[ind].reshape(28,28), cmap = matplotlib.cm.binary)
    plt.show()
    #plt.imshow(x_test_noisy[ind].reshape(16,15), cmap = matplotlib.cm.binary)
    #plt.show()
    plt.imshow(denoised_test[ind].reshape(28,28), cmap = matplotlib.cm.binary)
    plt.show()
#plt.imshow(x_train_noisy[12:12+1].reshape(28,28), cmap = matplotlib.cm.binary)
