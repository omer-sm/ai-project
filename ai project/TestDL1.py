import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from unit10 import c1w3_utils as u10
from DL1 import *

np.random.seed(1)
X, Y = u10.load_planar_dataset()
#plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);
#plt.show()
model = DLModel("points model")
model.add(DLLayer("l1", 10, (X.shape[0], ), "relu", alpha=1., optimization="adaptive"))
model.add(DLLayer("l3", 1, (10, ), "trim_sigmoid", alpha=1., optimization="adaptive"))
model.compile("cross_entropy")
costs = model.train(X,Y,10000)
u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = model.predict(X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')