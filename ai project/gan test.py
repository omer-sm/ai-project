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
x = np.arange(-1600,1600, 10).reshape(1,320)
x = np.concatenate((x, x**2+20), axis=0)

gen = DLModel()
gen.add(DLLayer("", 32, (16,), "leaky_relu", "He", 0.0005, "rmsprop"))
gen.add(DLLayer("", 64, (32,), "leaky_relu", "He", 0.0005, "rmsprop"))
gen.add(DLLayer("", 64, (64,), "leaky_relu", "He", 0.0005, "rmsprop"))
gen.add(DLLayer("", 64, (64,), "leaky_relu", "He", 0.0005, "rmsprop"))
gen.add(DLLayer("", 64, (64,), "leaky_relu", "He", 0.0005, "rmsprop"))
gen.add(DLLayer("", 2, (64,), "leaky_relu", "Xavier", 0.0005, "rmsprop"))
gen.compile("squared_means")
discrim = DLModel()
discrim.add(DLLayer("", 8, (2,), "leaky_relu", "Xavier", 0.0005))
discrim.add(DLLayer("", 16, (8,), "leaky_relu", "Xavier", 0.0005))
discrim.add(DLLayer("", 2, (16,), "leaky_relu", "Xavier", 0.0005))
discrim.add(DLLayer("", 1, (2,), "leaky_relu", "Xavier", 0.01))
discrim.compile("squared_means")
gan = DLGANModel("", gen, discrim)
gan.compile()
gan.train(x, 5000, 3, 5)
plt.plot(x[0], x[1])
y = gan.generate(32)
plt.scatter(y[0], y[1])
plt.show()
print(gan.generate(5))
exit()'''
digits = fetch_olivetti_faces()
digits = np.array(digits["data"])[:200].T
#digits = digits.reshape(1,64,1797)
digits *= 255.
digits = cp.array(digits)

gen = DLModel(use_cuda=True)
gen.add(DLLayer("", 128, (64,), "leaky_relu", "Xavier", 0.001))
gen.add(DLLayer("", 1024, (128,), "leaky_relu", "zeros", 0.001))
gen.add(DLLayer("", 512, (1024,), "leaky_relu", "Xavier", 0.001))
gen.add(DLLayer("", 4096, (512,), "leaky_relu", "zeros", 0.001))
'''gen.add(DLLayer("", 32, (128,), "leaky_relu", "Xavier", 0.01))
gen.add(DLLayer("", 64, (32,), "leaky_relu", "Xavier", 0.01))
l1 = DLDeflattenLayer("", (64,), (1,8,8))
gen.add(l1)
l2 = DLConvLayer("", 64, (1,8,8), "leaky_relu", "Xavier", 0.1, padding="same")
gen.add(l2)
gen.add(DLPixelShuffleLayer("", (64,8,8)))
l3 = DLConvLayer("", 16, (16,16,16), "leaky_relu", "Xavier", 0.01, padding="same")
gen.add(l3)
l3 = DLConvLayer("", 4, l3.get_output_shape(), "leaky_relu", "Xavier", 0.01, padding="same")
gen.add(l3)
l3 = DLPixelShuffleLayer("",l3.get_output_shape())
gen.add(l3)
l5 = DLUpsampleLayer("", l3.get_output_shape())
gen.add(l5)
gen.add(DLFlattenLayer("", l5.get_output_shape()))'''
gen.compile("squared_means")

discrim = DLModel(use_cuda=True)
discrim.add(DLLayer("", 64, (4096,), "leaky_relu", "Xavier", 0.001))
discrim.add(DLLayer("", 256, (64,), "leaky_relu", "He", 0.001))
#discrim.add(DLLayer("", 32, (256,), "leaky_relu", "He"))
'''l1 = DLDeflattenLayer("", (4096,), (1,64,64))
discrim.add(l1)
l = DLConvLayer("", 16, (1,64,64), "leaky_relu", "Xavier", 0.01, (5,5), (2,2), padding="valid")
discrim.add(l)
l = DLConvLayer("", 64, l.get_output_shape(), "leaky_relu", "Xavier", 0.01, (3,3), (2,2), padding="valid")
discrim.add(l)
l2 = DLConvLayer("", 2, l.get_output_shape(), "leaky_relu", "Xavier", 0.01, padding="valid")
discrim.add(l2)
l3 = DLMaxPoolingLayer("", l2.get_output_shape())
discrim.add(l3)
l4 = DLFlattenLayer("", l3.get_output_shape())
discrim.add(l4)
discrim.add(DLLayer("", 32, l4.get_output_shape(), "leaky_relu", "Xavier", 0.01))'''
#discrim.add(DLLayer("", 256, (32,), "leaky_relu", "Xavier", 0.01))
discrim.add(DLLayer("", 1, (256,), "trim_sigmoid", "Xavier", 0.001))
discrim.compile("cross_entropy")

gan = DLGANModel("", gen, discrim)
gan.compile()

costs = gan.train(digits, 80, 3, 5)

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