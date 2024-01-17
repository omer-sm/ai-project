from DL10 import *
import numpy as np
#import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
import os, os.path
from PIL import Image
from sklearn.datasets import load_digits, fetch_olivetti_faces

np.random.seed(1)
#cp.random.seed(1)

#x = np.array([[-3,9],[-1.5, 2.25], [3, 9], [4,40], [5,50], [6,60]]).T
x = np.arange(-30,30, 5).reshape(1,12)
x = np.concatenate((x, x**2+1), axis=0)

gen = DLModel()
gen.add(DLLayer("", 256, (2,), "leaky_relu", "He", 6e-4, "rmsprop"))
gen.add(DLLayer("", 16, (256,), "vae_bottleneck", "He", 6e-4, "rmsprop", samples_per_dim=4))
gen.add(DLLayer("", 64, (32,), "leaky_relu", "He", 6e-4, "rmsprop"))
l = DLLayer("", 2, (64,), "leaky_relu", "Xavier", 6e-4, "rmsprop")
l.leaky_relu_d = 1.
gen.layers[1].leaky_relu_d = 1.
gen.add(l)
gen.compile("squared_means_KLD")
discrim = DLModel()
discrim.add(DLLayer("", 8, (2,), "leaky_relu", "Xavier", 6e-4, "rmsprop"))
discrim.add(DLLayer("", 16, (8,), "leaky_relu", "Xavier", 6e-4, "rmsprop"))
discrim.add(DLLayer("", 32, (16,), "leaky_relu", "Xavier", 6e-4, "rmsprop"))
discrim.add(DLLayer("", 1, (32,), "trim_sigmoid", "Xavier", 6e-4, "rmsprop"))
discrim.compile("squared_means")
gan = DLVAEGANModel("", gen, discrim)
gan.compile()
gan.train(x, 5000, 1, 5, 3)
plt.plot(x[0], x[1])
y = gan.generate(100)
plt.scatter(y[0], y[1])
plt.show()
print(gan.generate(5))
exit()
digits = fetch_olivetti_faces()
digits = np.array(digits["data"])[:200].T
#digits = digits.reshape(1,64,1797)
#digits *= 255.
digits = cp.array(digits)

gen = DLModel(use_cuda=True)
#gen.add(DLLayer("", 128, (64,), "leaky_relu", "Xavier", 1e-3, "rmsprop"))
gen.add(DLLayer("", 512, (128,), "leaky_relu", "Xavier", 1e-3, "rmsprop"))
gen.add(DLLayer("", 32, (512,), "leaky_relu", "Xavier", 1e-3, "rmsprop"))
gen.add(DLLayer("", 64, (32,), "leaky_relu", "Xavier", 1e-3, "rmsprop"))
l1 = DLDeflattenLayer("", (64,), (1,8,8))
gen.add(l1)
l2 = DLConvLayer("", 32, (1,8,8), "leaky_relu", "Xavier", 1e-3, padding="same", optimization="rmsprop")
gen.add(l2)
gen.add(DLPixelShuffleLayer("", (32,8,8)))
l3 = DLConvLayer("", 16, (8,16,16), "leaky_relu", "Xavier", 1e-3, padding="same", optimization="rmsprop")
gen.add(l3)
l3 = DLConvLayer("", 4, l3.get_output_shape(), "leaky_relu", "Xavier", 1e-3, padding="same", optimization="rmsprop")
gen.add(l3)
l3 = DLPixelShuffleLayer("",l3.get_output_shape())
gen.add(l3)
l5 = DLUpsampleLayer("", l3.get_output_shape())
gen.add(l5)
l6 = DLConvLayer("", 8, l5.get_output_shape(), "leaky_relu", "Xavier", 1e-3, padding="same", optimization="rmsprop")
gen.add(l6)
l6 = DLConvLayer("", 8, l6.get_output_shape(), "leaky_relu", "Xavier", 1e-3, padding="same", optimization="rmsprop")
gen.add(l6)
l6 = DLConvLayer("", 1, l6.get_output_shape(), "trim_sigmoid", "Xavier", 1e-3, padding="same", optimization="rmsprop")
gen.add(l6)
gen.add(DLFlattenLayer("", l6.get_output_shape()))
gen.compile("squared_means")

discrim = DLModel(use_cuda=True)
#discrim.add(DLLayer("", 64, (4096,), "leaky_relu", "Xavier", 5e-5, optimization="rmsprop"))
#discrim.add(DLLayer("", 32, (64,), "leaky_relu", "He", 5e-5, optimization="rmsprop"))
#discrim.add(DLLayer("", 4096, (32,), "leaky_relu", "He", 5e-5, optimization="rmsprop"))
l1 = DLDeflattenLayer("", (4096,), (1,64,64))
discrim.add(l1)
l = DLConvLayer("", 8, (1,64,64), "leaky_relu", "Xavier", 1e-3, (5,5), (2,2), padding="valid", optimization="rmsprop")
discrim.add(l)
l = DLConvLayer("", 16, l.get_output_shape(), "leaky_relu", "Xavier", 1e-3, (3,3), (2,2), optimization="rmsprop", padding="valid")
discrim.add(l)
l2 = DLConvLayer("", 8, l.get_output_shape(), "leaky_relu", "Xavier", 1e-3, (3,3), (2,2), optimization="rmsprop", padding="same")
discrim.add(l2)
l3 = DLMaxPoolingLayer("", l2.get_output_shape())
discrim.add(l3)
l4 = DLFlattenLayer("", l3.get_output_shape())
discrim.add(l4)
discrim.add(DLLayer("", 32, l4.get_output_shape(), "leaky_relu", "Xavier", 1e-3, optimization="rmsprop"))
discrim.add(DLLayer("", 256, (32,), "leaky_relu", "Xavier", 1e-3, optimization="rmsprop"))
discrim.add(DLLayer("", 1, (256,), "leaky_relu", "Xavier", 1e-3, optimization="rmsprop"))
discrim.compile("cross_entropy")

gan = DLGANModel("", gen, discrim)
gan.compile()

costs = gan.train(digits, 400, 2, 5)

print("showing gen")
#print(gan._generate_noise(4))
for i in range(20):
    out = cp.asnumpy(gan.generate(20).T)
    print(out)
    print(np.max(out))
    plt.imshow(out[i].reshape(64,64,1), cmap="gray")
    plt.show()
plt.imshow(cp.asnumpy(digits.T.reshape(200,64,64,1))[0], cmap="gray")
plt.show()
exit()
imgs = []
path = r"C:\Users\omerg\Favorites\Downloads\cats"
for i in range(1):
    for f in os.listdir(path)[i*300:(i+1)*300]:
        ext = os.path.splitext(f)[1]
        if ext.lower() != ".jpg":
            continue
        img = np.array(Image.open(os.path.join(path,f)))
        imgs.append(img)
imgs = cp.array(imgs)[:20].T /255.
imgs_flat = imgs.T.reshape(20,4096*3).T
print("loaded data")

gen = DLModel(use_cuda=True)
gen.add(DLLayer("", 72, (20,), "leaky_relu", "Xavier", 1.1))
l1 = DLDeflattenLayer("", (72,), (2,6,6))
gen.add(l1)
l2 = DLConvLayer("", 48, l1.get_output_shape(), "leaky_relu", "He", 1.1, (3,3), padding=(2,2), strides=(1,1))
gen.add(l2)
l3 = DLPixelShuffleLayer("", l2.get_output_shape())
gen.add(l3)
l4 = DLPixelShuffleLayer("", l3.get_output_shape())
gen.add(l4)
l5 = DLConvLayer("", 32, l4.get_output_shape(), "relu", "He", 1.1, (3,3), padding=(1,1), strides=(1,1))
gen.add(l5)
l6 = DLConvLayer("", 3, l5.get_output_shape(), "trim_sigmoid", "He", 1.1, padding=(1,1), strides=(1,1))
gen.add(l6)
l7 = DLUpsampleLayer("", l6.get_output_shape())
gen.add(l7)
l8 = DLFlattenLayer("", l7.get_output_shape())
gen.add(l8)


discrim = DLModel(use_cuda=True)
l1 = DLDeflattenLayer("", (4096*3,), (3,64,64))
discrim.add(l1)
l2 = DLConvLayer("", 8, (3,64,64), "leaky_relu", "Xavier", 0.01, (5,5), (4,4), "valid")
discrim.add(l2)
l3 = DLMaxPoolingLayer("", l2.get_output_shape(), (3,3))
discrim.add(l3)
l4 = DLConvLayer("", 8, l3.get_output_shape(), "relu", "He", 0.01, (3,3), (2,2), "valid")
discrim.add(l4)
l5 = DLMaxPoolingLayer("", l4.get_output_shape(), (3,3))
discrim.add(l5)
l6 = DLFlattenLayer("", l5.get_output_shape())
discrim.add(l6)
l7 = DLLayer("", 128, l6.get_output_shape(), "leaky_relu", "Xavier", 0.01)
discrim.add(l7)
l8 = DLLayer("", 64, (128,), "leaky_relu", "He", 0.01)
discrim.add(l8)
l9 = DLLayer("", 1, (64,), "trim_sigmoid", "Xavier", 0.01)
discrim.add(l9)

gen.compile("squared_means")
discrim.compile("cross_entropy")

gan = DLGANModel("", gen, discrim)
gan.compile()

costs = gan.train(imgs_flat, 80, 5, 5, 10)

print("showing gen")
#print(gan._generate_noise(4))
for i in range(20):
    out = cp.asnumpy(gan.generate(20).T)
    print(out)
    plt.imshow(out[i].reshape(64,64,3), cmap=matplotlib.cm.binary)
    plt.show()