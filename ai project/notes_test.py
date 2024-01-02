import h5py
import numpy as np
import cupy as cp
from DL7 import *
from matplotlib import pyplot as plt

f = h5py.File(r"C:\Users\omerg\source\repos\note sample generation\note sample generation\note_samples.hdf5", "r")
X_total = []
Y_total = np.array([])

for note in range(len(list(f.keys()))):
    dset = np.array(f[list(f.keys())[note]])
    X_total = dset if len(X_total) == 0 else np.concatenate((X_total, dset))
    Y_total = np.append(Y_total, [note] * 200)

np.random.seed(1)
np.random.shuffle(X_total)
np.random.seed(1)
np.random.shuffle(Y_total)

X_train = X_total[0:1800]
X_train = np.array(X_train).T
Y_train = Y_total[0:1800]
Y_train = DLModel.to_one_hot(12, Y_train)
X_test = X_total[1800:]
X_test = np.array(X_test).T
Y_test = Y_total[1800:]
Y_test = DLModel.to_one_hot(12, Y_test)
n = X_train.shape[0]

X_train = cp.array(X_train)
Y_train = cp.array(Y_train)
X_test = cp.array(X_test)
Y_test = cp.array(Y_test)

np.random.seed(1)
cp.random.seed(1)
model = DLModel(use_cuda=True)
model.add(DLLayer("L1", 100, (n,), "relu", "He", 0.01, "adam"))
model.add(DLLayer("L2", 40, (100,), "leaky_relu", "Xavier", 0.01, "adam"))
model.add(DLLayer("L3", 12, (40,), "trim_softmax", "Xavier", 0.01, "adam"))
model.compile("categorical_cross_entropy")
costs = model.train(X_train, Y_train, 100, 600)

plt.plot(costs)
plt.show()

print("Train:")
model.confusion_matrix(X_train, Y_train)
print("Test:")
model.confusion_matrix(X_test, Y_test)