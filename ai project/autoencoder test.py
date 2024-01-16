from DL9 import *
import cupy as cp
import numpy as np
from sklearn.datasets import fetch_openml, load_digits
import matplotlib.pyplot as plt
import matplotlib
import os, os.path
from PIL import Image

'''np.random.seed(1)
mnist = fetch_openml('mfeat-pixel',as_frame=False, parser="liac-arff")
digits = mnist["data"]
y = digits.T
x = digits.reshape(2000,16,15).T
x = np.array([x])
x *= 255/6
y *= 255/6
y = cp.array(y)
x = cp.array(x)
model = DLModel("", use_cuda=True)
l1 = DLConvLayer("", 32, (1,15,16), "leaky_relu", "Xavier", 0.001, (4,4), (2,1), "same", "adam")
model.add(l1)
l2 = DLConvLayer("", 32, l1.get_output_shape(), "relu", "Xavier", 0.001, (3,3), (1,2), "valid", "adam")
model.add(l2)
l3 = DLConvLayer("", 16, l2.get_output_shape(), "leaky_relu", "He", 0.001, (2,2), (1,1), "valid", "adam")
model.add(l3)
l4 = DLMaxPoolingLayer("", l3.get_output_shape(), (2,2), (1,1))
model.add(l4)
l5 = DLFlattenLayer("", l4.get_output_shape())
model.add(l5)
print(l5.get_output_shape())
model.add(DLLayer("", 512, l5.get_output_shape(), "leaky_relu", "He", 0.001, "adam"))
model.add(DLLayer("", 32, (512,), "vae_bottleneck", "Xavier", 0.002, "adaptive", samples_per_dim=4))
model.add(DLLayer("", 512, (64,), "relu", "Xavier", 0.001, "adam"))
model.add(DLLayer("", 240, (512,), "leaky_relu", "Xavier", 0.001, "adam"))
model.compile("squared_means_KLD", recon_loss_weight=1.1, KLD_beta=0.4)
model.train(x, y, 100, 500)
processed = model.forward_propagation(x)
processed = processed.reshape(16,15,2000).transpose(2,0,1)
processed = cp.asnumpy(processed)
x = cp.asnumpy(x)
for i in range(20):
    plt.imshow(x.T[i*30+194], cmap="gray")
    plt.show()
    plt.imshow(processed[i*30+194], cmap="gray")
    plt.show()
    
exit()'''

np.random.seed(1)
cp.random.seed(1)

imgs = []
path = r"C:\Users\omerg\Favorites\Downloads\cats"
for i in range(1):
    for f in os.listdir(path)[i*300:(i+1)*300]:
        ext = os.path.splitext(f)[1]
        if ext.lower() != ".jpg":
            continue
        img = np.array(Image.open(os.path.join(path,f)))
        imgs.append(img)
imgs = cp.array(imgs)[90:190].T
imgs_flat = imgs.T.reshape(100,4096*3).T
imgs = imgs
print("loaded data")
imgs = cp.asnumpy(imgs.T)[:20]
'''im1 = imgs[:5]
im2 = imgs[5:10]
im3 = imgs[10:15]
im4 = imgs[15:20]
comb = np.insert(im1, np.arange(64), im2, axis=2)
comb2 = np.insert(im3, np.arange(64), im4, axis=2)
ret = np.insert(comb, np.arange(64), comb2, axis=1)
print(ret.shape)'''
l = DLPixelShuffleLayer("", (20,64,64))
ret = l.forward_propagation(imgs)
ret = l.backward_propagation(ret)
plt.imshow(imgs[17])
plt.show()
plt.imshow(ret[17])
plt.show()
exit()
model = DLModel("", use_cuda=True)

l1 = DLConvLayer("", 32, (3,64,64), "relu", "He", 0.0003, (4,4), (3,3), "valid", "adam")
model.add(l1)
l2 = DLConvLayer("", 16, l1.get_output_shape(), "leaky_relu", "Xavier", 0.0005, (3,3), (2,2), (1,1), "adam")
model.add(l2)
l3 = DLMaxPoolingLayer("", l2.get_output_shape(), strides=(2,2))
model.add(l3)
l4 = DLConvLayer("", 32, l3.get_output_shape(), "relu", "He", 0.0003, (3,3), (2,2), "valid", "adam")
model.add(l4)
l5 = DLConvLayer("", 16, l4.get_output_shape(), "leaky_relu", "Xavier", 0.0005, strides=(2,2), optimization="adam")
model.add(l5)
l6 = DLMaxPoolingLayer("", l5.get_output_shape(), (2,2))
model.add(l6)
l7 = DLFlattenLayer("", l6.get_output_shape())
model.add(l7)
model.add(DLLayer("", 512, l7.get_output_shape(), "leaky_relu", "Xavier", 0.001, "adam")) 
model.add(DLLayer("", 16, (512,), "vae_bottleneck", "Xavier", 0.002, "adaptive", samples_per_dim=16)) 
model.add(DLLayer("", 2048, (128,), "leaky_relu", "He", 0.001, "adam"))
model.add(DLLayer("", 4096*3, (2048,), "relu", "Xavier", 0.001, "adam"))
model.compile("squared_means_KLD", recon_loss_weight=1.1, KLD_beta=1.5)
costs = model.train(imgs, imgs_flat, 1500, 25)
processed = (cp.asnumpy(model.forward_propagation(imgs)).T).astype(int)
imgs = cp.asnumpy(imgs.T ).astype(int)
model.save_weights("cat_vae", True)
for i in range(5):
    ind = i*np.random.randint(1,2)
    plt.imshow(imgs[ind].reshape(64,64,3))
    plt.show()
    plt.imshow(processed[ind].reshape(64,64,3))
    plt.show()

print("showing gen")

for i in range(20):
    gen = np.random.random_sample((16, 1))*(10-i)/10
    gen = cp.array(gen)
    gen = model.layers[9]._vae_bottleneck(gen)
    for l in range(10,12):
        gen = model.layers[l].forward_propagation(gen, False)
    gen = cp.asnumpy(gen.T).astype(int)
    plt.imshow(gen.reshape(64,64,3))
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
