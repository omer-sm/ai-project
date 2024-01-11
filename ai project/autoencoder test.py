from DL9 import *
import cupy as cp
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib
import os, os.path
from PIL import Image

np.random.seed(1)
cp.random.seed(1)

imgs = []
path = r"C:\Users\omerg\Favorites\Downloads\cats"
for i in range(1):
    for f in os.listdir(path)[i*3000:(i+1)*3000]:
        ext = os.path.splitext(f)[1]
        if ext.lower() != ".jpg":
            continue
        img = np.array(Image.open(os.path.join(path,f)).convert("L"))
        imgs.append(img)
imgs = cp.array(imgs)[:3000].T
imgs_flat = imgs.T.reshape(3000,4096).T
imgs = imgs.reshape(1,64,64,3000)
imgs = imgs / 255.

model = DLModel("", use_cuda=True)
model.add(l1 := DLConvLayer("1", 8, (1,64,64), "trim_tanh", "Xavier", 0.002, (5,5), (3,3), "valid", optimization="adam"))
model.add(l2 := DLMaxPoolingLayer("2", l1.get_output_shape(), (4,4), (3,3)))
#model.add(l3 := DLConvLayer("3", 8, l2.get_output_shape(), "trim_tanh", "Xavier", 0.002, (4,4), (3,3), "valid", optimization="adam"))
#model.add(l4 := DLMaxPoolingLayer("4", l3.get_output_shape(), (3,3), (1,1)))
model.add(l5 := DLFlattenLayer("5", l2.get_output_shape()))
model.add(DLLayer("2", 8, l5.get_output_shape(), "vae_bottleneck", "Xavier", 0.001, "adam", samples_per_dim=4))
model.add(DLLayer("3", 64, (16,), "trim_tanh", "He", 0.002, "adam"))
model.add(DLLayer("4", 4096, (64,), "trim_sigmoid", "Xavier", 0.002, "adam"))
model.compile("cross_entropy_KLD", recon_loss_weight=0.8)
costs = model.train(imgs, imgs_flat, 50, 1000)
processed = (cp.asnumpy(model.forward_propagation(imgs)).T)
imgs = cp.asnumpy(imgs.T )

for i in range(20):
    ind = i*np.random.randint(1,100)
    plt.imshow(imgs[ind].reshape(64,64), cmap=matplotlib.cm.binary)
    plt.show()
    plt.imshow(processed[ind].reshape(64,64), cmap=matplotlib.cm.binary)
    plt.show()

'''mnist = fetch_openml('mnist_784',as_frame=False, parser="liac-arff")
digits = mnist["data"]
m = 6000
x_train, x_test = digits[:m], digits[m:m+1000]
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

x_train_noisy = x_train_noisy.reshape(1,28,28,6000)
x_test_noisy = x_test_noisy.reshape(1,28,28,1000)

model = DLModel("", use_cuda=True)
model.add(l1 := DLConvLayer("1", 6, (1,28,28), "trim_tanh", "He", 0.002, (5,5), (4,4), optimization="adam"))
model.add(l2 := DLMaxPoolingLayer("2", l1.get_output_shape(), (4,4), (3,3)))
model.add(l3 := DLConvLayer("3", 4, l2.get_output_shape(), "relu", "Xavier", 0.002, strides=(2,2), optimization="adam"))
model.add(l4 := DLMaxPoolingLayer("4", l3.get_output_shape(), strides=(2,2)))
model.add(l5 := DLFlattenLayer("5", l4.get_output_shape()))
model.add(DLLayer("6", 16, l5.get_output_shape(), "vae_bottleneck", "He", 0.002, "adam", samples_per_dim=6))
model.add(DLLayer("7", 128, (48,), "leaky_relu", "He", 0.002, "adam"))
model.add(DLLayer("8", 784, (128,), "trim_sigmoid", "Xavier", 0.002, "adam"))
model.compile("cross_entropy_KLD", recon_loss_weight=0.8)
costs = model.train(x_train_noisy, x_train, 200, 1200)
costs = cp.asnumpy(costs)
plt.plot(costs)
plt.show()
encoded = cp.array(digits[::1000]).reshape(1,28,28,70)
for l in range(1,9):
    encoded = model.layers[l].forward_propagation(encoded, False)
encoded = cp.asnumpy(encoded)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(encoded[0] / np.max(np.abs(encoded[0])), encoded[1] / np.max(np.abs(encoded[1])), encoded[2] / np.max(np.abs(encoded[2])), c=np.array(mnist["target"])[::1000].astype(int))
plt.show()
gen = cp.random.normal(0.5, 0.5, (48,4))
for l in range(7,9):
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
#plt.imshow(x_train_noisy[12:12+1].reshape(28,28), cmap = matplotlib.cm.binary)'''
