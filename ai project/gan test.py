from DL10 import *
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
import os, os.path
from PIL import Image
from sklearn.datasets import load_digits, fetch_olivetti_faces

np.random.seed(1)
cp.random.seed(1)

'''#x = np.array([[-3,9],[-1.5, 2.25], [3, 9], [4,40], [5,50], [6,60]]).T
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
exit()'''
'''digits = fetch_olivetti_faces()
digits = np.array(digits["data"])[:200].T
#digits = digits.reshape(1,64,1797)
#digits *= 255.
digits = cp.array(digits)

gen = DLModel(use_cuda=True)
l = DLDeflattenLayer("", (4096,), (1,64,64))
gen.add(l)
l = DLConvLayer("", 12, l.get_output_shape(), "leaky_relu", "Xavier", 1e-3, strides=(2,2), padding=(1,1), optimization="rmsprop")
gen.add(l)
l = DLMaxPoolingLayer("", l.get_output_shape())
gen.add(l)
l = DLConvLayer("", 12, l.get_output_shape(), "leaky_relu", "Xavier", 1e-3, strides=(2,2), padding=(1,1), optimization="rmsprop")
gen.add(l)
l = DLMaxPoolingLayer("", l.get_output_shape())
gen.add(l)
l = DLConvLayer("", 8, l.get_output_shape(), "leaky_relu", "Xavier", 1e-3, strides=(2,2), padding=(1,1), optimization="rmsprop")
gen.add(l)
l = DLMaxPoolingLayer("", l.get_output_shape())
gen.add(l)
l = DLFlattenLayer("", l.get_output_shape())
gen.add(l)
l = DLLayer("", 8, l.get_output_shape(), "vae_bottleneck", "Xavier", 1e-3, "rmsprop", samples_per_dim=8)
gen.add(l)
l = DLLayer("", 64, l.get_output_shape(), "leaky_relu", "Xavier", 1e-3, "rmsprop")
gen.add(l)
l = DLDeflattenLayer("", l.get_output_shape(), (1,8,8))
gen.add(l)
l = DLConvLayer("", 12, l.get_output_shape(), "leaky_relu", "Xavier", 1e-3, strides=(1,1), padding=(1,1), optimization="rmsprop")
gen.add(l)
l = DLPixelShuffleLayer("", l.get_output_shape())
gen.add(l)
l = DLConvLayer("", 4, l.get_output_shape(), "leaky_relu", "Xavier", 1e-3, strides=(1,1), padding=(1,1), optimization="rmsprop")
gen.add(l)
l = DLUpsampleLayer("", l.get_output_shape())
gen.add(l)
l = DLPixelShuffleLayer("", l.get_output_shape())
gen.add(l)
l = DLConvLayer("", 12, l.get_output_shape(), "leaky_relu", "Xavier", 1e-3, strides=(1,1), padding="same", optimization="rmsprop")
gen.add(l)
l = DLConvLayer("", 1, l.get_output_shape(), "leaky_relu", "Xavier", 1e-3, strides=(1,1), padding="same", optimization="rmsprop")
l.leaky_relu_d = 1.
gen.add(l)
gen.add(DLFlattenLayer("", l.get_output_shape()))
#print(l.get_output_shape())
gen.compile("squared_means_KLD")

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

gan = DLVAEGANModel("", gen, discrim)
gan.compile()

costs = gan.train(digits, 250, 2, 5, 50)

print("showing gen")
#print(gan._generate_noise(4))
for i in range(20):
    out = cp.asnumpy(gan.generate(20).T)
    #print(out)
    #print(np.max(out))
    plt.imshow(out[i].reshape(64,64,1), cmap="gray")
    plt.show()
plt.imshow(cp.asnumpy(digits.T.reshape(200,64,64,1))[0], cmap="gray")
plt.show()
exit()'''
imgs = []
path = r"C:\Users\omerg\Favorites\Downloads\cats"
for i in range(2):
    for f in os.listdir(path)[i*300:(i+1)*300]:
        ext = os.path.splitext(f)[1]
        if ext.lower() != ".jpg":
            continue
        img = np.array(Image.open(os.path.join(path,f)))
        imgs.append(img)
imgs = cp.array(imgs)[:4].T /255.
imgs_flat = imgs.T.reshape(4,4096*3).T
print("loaded data")

gen = DLModel(use_cuda=True)
l = DLDeflattenLayer("", (4096*3,), (3,64,64))
gen.add(l)
l = DLConvLayer("", 32, l.get_output_shape(), "leaky_relu", "Xavier", 3e-3, strides=(2,2), padding=(1,1), optimization="rmsprop")
gen.add(l)
l = DLMaxPoolingLayer("", l.get_output_shape())
gen.add(l)
l = DLConvLayer("", 32, l.get_output_shape(), "leaky_relu", "Xavier", 3e-3, strides=(2,2), padding=(1,1), optimization="rmsprop")
gen.add(l)
l = DLMaxPoolingLayer("", l.get_output_shape())
gen.add(l)
l = DLConvLayer("", 16, l.get_output_shape(), "leaky_relu", "Xavier", 3e-3, strides=(2,2), padding=(1,1), optimization="rmsprop")
gen.add(l)
l = DLMaxPoolingLayer("", l.get_output_shape())
gen.add(l)
l = DLFlattenLayer("", l.get_output_shape())
gen.add(l)
l = DLLayer("", 12, l.get_output_shape(), "vae_bottleneck", "Xavier", 1e-2, "rmsprop", samples_per_dim=8)
gen.add(l)
l = DLLayer("", 128, l.get_output_shape(), "leaky_relu", "Xavier", 1e-2, "rmsprop")
gen.add(l)
l = DLLayer("", 64, l.get_output_shape(), "leaky_relu", "Xavier", 3e-3, "rmsprop")
gen.add(l)
l = DLDeflattenLayer("", l.get_output_shape(), (1,8,8))
gen.add(l)
l = DLConvLayer("", 24, l.get_output_shape(), "leaky_relu", "Xavier", 3e-3, strides=(1,1), padding=(1,1), optimization="rmsprop")
gen.add(l)
l = DLPixelShuffleLayer("", l.get_output_shape())
gen.add(l)
l = DLConvLayer("", 32, l.get_output_shape(), "leaky_relu", "Xavier", 3e-3, strides=(1,1), padding=(1,1), optimization="rmsprop")
gen.add(l)
#l = DLUpsampleLayer("", l.get_output_shape())
l = DLPixelShuffleLayer("", l.get_output_shape())
gen.add(l)
l = DLPixelShuffleLayer("", l.get_output_shape())
gen.add(l)
l = DLConvLayer("", 12, l.get_output_shape(), "leaky_relu", "zeros", 3e-3, strides=(1,1), padding="same", optimization="rmsprop")
gen.add(l)
l = DLConvLayer("", 3, l.get_output_shape(), "trim_sigmoid", "Xavier", 3e-3, strides=(1,1), padding="same", optimization="rmsprop")
l.leaky_relu_d = 1.
gen.add(l)
gen.add(DLFlattenLayer("", l.get_output_shape()))
#print(l.get_output_shape())
gen.compile("squared_means_KLD", KLD_beta=0.8)
gen.train(imgs_flat, imgs_flat, 25)

discrim = DLModel(use_cuda=True)
#discrim.add(DLLayer("", 64, (4096,), "leaky_relu", "Xavier", 5e-5, optimization="rmsprop"))
#discrim.add(DLLayer("", 32, (64,), "leaky_relu", "He", 5e-5, optimization="rmsprop"))
#discrim.add(DLLayer("", 4096, (32,), "leaky_relu", "He", 5e-5, optimization="rmsprop"))
l1 = DLDeflattenLayer("", (4096*3,), (3,64,64))
discrim.add(l1)
l = DLConvLayer("", 8, (3,64,64), "leaky_relu", "Xavier", 3e-3, (5,5), (2,2), padding="valid", optimization="rmsprop")
discrim.add(l)
l = DLConvLayer("", 16, l.get_output_shape(), "leaky_relu", "Xavier", 3e-3, (3,3), (2,2), optimization="rmsprop", padding="valid")
discrim.add(l)
l2 = DLConvLayer("", 8, l.get_output_shape(), "leaky_relu", "Xavier", 3e-3, (3,3), (2,2), optimization="rmsprop", padding="same")
discrim.add(l2)
l3 = DLMaxPoolingLayer("", l2.get_output_shape())
discrim.add(l3)
l4 = DLFlattenLayer("", l3.get_output_shape())
discrim.add(l4)
discrim.add(DLLayer("", 32, l4.get_output_shape(), "leaky_relu", "Xavier", 3e-3, optimization="rmsprop"))
discrim.add(DLLayer("", 256, (32,), "leaky_relu", "Xavier", 3e-3, optimization="rmsprop"))
discrim.add(DLLayer("", 1, (256,), "leaky_relu", "Xavier", 3e-3, optimization="rmsprop"))
discrim.compile("cross_entropy")

gan = DLVAEGANModel("", gen, discrim)
gan.compile("wasserstein", 0.9)

costs = gan.train(imgs_flat, 50, 2, 5, 0, True)

print("showing gen")
#print(gan._generate_noise(4))
out = cp.asnumpy(gan.generate(20).T)
print(out)
fig, axs = plt.subplots(4, 5, figsize=(10, 8))
# Flatten the 2D array of subplots to a 1D array
axs = axs.flatten()
for i in range(20):
    axs[i].imshow(out[i].reshape(64, 64, 3), cmap=matplotlib.cm.binary)
    axs[i].axis('off')  # Turn off axis labels to improve visualization

plt.show()

