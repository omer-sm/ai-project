import numpy as np
from matplotlib import pyplot as plt
from unit10 import b_utils
from unit10 import c1w2_utils as c1w2
import scipy
from scipy import ndimage
from scipy.ndimage import zoom
from PIL import Image
import h5py
import random as rnd

def main2():
    X, Y = np.array([[1.,2.,-1., 4., 1., 5.],[3.,4.,-3.2, 3., 6., -4.2,]]), np.array([1,0,1, 0, 0, 1])
    print (X.shape, Y.shape)
    #W, b = train_logical_unadaptive(X, Y, 50000, 0.009, True)

    #print ("W = " + str(W))

    #print ("b = " + str(b))

def main2():
    x = np.array([[1,4], [2,2], [5, 8], [3, 4], [3, 9], [15, 2]])
    y = np.array([1, 1, 0, 0, 0, 0])
    x = x.reshape(x.shape[0], -1).T
    y = y.reshape(y.shape[0], -1)
    a = np.linspace(-10, 10)
    plt.plot(range(len(a)), sigmoid(a))
    plt.show()
    

    #train_logical_unadaptive(x, y, 4000, 0.005, True)

def main():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = c1w2.load_datasetC1W2()
    m_train = len(train_set_y)
    m_test = len(test_set_y)
    num_px = len(train_set_x_orig[0])
    train_set_x_flatten = (train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)).T
    test_set_x_flatten = (test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)).T
    train_set_x = train_set_x_flatten/255.0
    test_set_x = test_set_x_flatten/255.0
    W, b = train_logical_adaptive(train_set_x, train_set_y, 3000, 0.005)
    predictImage(W, b, r"C:\Users\omerg\Favorites\Downloads\Gym-structure-1080x675.png")

#vectors
class vector:
    def __init__ (self, size, isCol=True, fill=0, initVals=None):
        self.isCol = isCol
        self.vals = []
        self.size = size
        if initVals == None:
            for i in range(size):
                self.vals.append(fill)
        if initVals != None:
            for i in range(size):
                self.vals.append(initVals[i%len(initVals)])

    def __str__ (self):
        ret = "["
        ending = (",\n" if self.isCol else ", ")
        for v in self.vals:
            ret += str(v) + ending
        ret = ret[:-2]
        ret += "]"
        return ret

    def transpose(self):
        return vector (self.size, not self.isCol, initVals = self.vals)

    def dot(self, other):
        if self.size != other.size:
            raise Exception("vectors must be the same size for dot product.")
        if self.isCol or (not other.isCol):
            raise Exception("vectors need to be a row and a column (respectively) for dot product.")
        ret = 0
        tempVector = self * other.transpose()
        for i in tempVector:
            ret += i
        return ret


    def __checkCompatibility(self, other):
      if self.size != other.size:
        raise Exception("vectors must be the same size for math operations.")
      if self.isCol != other.isCol:
        raise Exception("vectors must both be the same axis for math operations.")

    def __add__(self, other):
        retVals = []
        if isinstance(other, vector):
            self.__checkCompatibility(other)
            for i in range(self.size):
                retVals.append(self.vals[i] + other.vals[i])
        else:
            for i in range(self.size):
                retVals.append(self.vals[i] + other)
        return vector(self.size, self.isCol, initVals = retVals)
    
    def __sub__(self, other):
        return self + other * (-1)

    def __mul__(self, other):
        retVals = []
        if isinstance(other, vector):
            self.__checkCompatibility(other)
            for i in range(self.size):
                retVals.append(self.vals[i] * other.vals[i])
        else:
            for i in range(self.size):
                retVals.append(self.vals[i] * other)
        return vector(self.size, self.isCol, initVals = retVals)

    def __truediv__(self, other):
        retVals = []
        if isinstance(other, vector):
            self.__checkCompatibility(other)
            for i in range(self.size):
                retVals.append(self.vals[i] / other.vals[i])
        else:
            for i in range(self.size):
                retVals.append(self.vals[i] / other)
        return vector(self.size, self.isCol, initVals = retVals)

    def __getitem__(self, key):
        return self.vals[key]

    def __setitem__(self, key, value):
        self.vals[key] = value

    def __len__(self):
        return len(self.vals)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self * (-1) + other

    def __rtruediv__(self, other):
        return vector(self.size, self.isCol, fill=other) / self
    
    def __rmul__(self, other):
        return self * other

    def __lt__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] < other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

    def __le__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] <= other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

    def __eq__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] == other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

    def __ne__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] != other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

    def __gt__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] > other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

    def __ge__(self, other):
        retVals = []
        for i in range(self.size):
            retVals.append( 1 if self[i] >= other else 0)
        return vector(self.size, self.isCol, initVals = retVals)

#logical regression
def predict(X, W, b):
    Z = np.dot(X.T, W)+b
    A = sigmoid(Z).T
    print(A)
    return np.where(A > 0.5, 1, 0)

def predictImage(W, b, path):
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize((64, 64), Image.LANCZOS)
    my_image = np.array(img).reshape(1, -1).T
    my_predicted_image = predict(my_image, W, b)
    print(my_predicted_image)




def sigmoid(z):
    temp = 1/(1+np.exp(-z)) 
    for t in range(len(temp)):
        if (temp[t] == 0):
            temp[t] = 0.000001
        elif (temp[t] == 1):
            temp[t] = 0.999999
    return temp

def initialize_with_zeros(dim):
    return np.zeros((dim, 1)), 0.01

def forward_propagation(X, Y, w, b):
    Z = np.dot(X.T, w)+b
    A = sigmoid(Z).T
    #L = Y*np.log(A) + (1-Y)*np.log(1-A)
    L = np.log(np.abs(np.bitwise_xor(Y, 1)-A))
    J = -np.average(L)
    return A, J

def backward_propagation(X, Y, A):
    dZ = (A-Y.T)/len(Y)
    dW = np.dot(X, dZ.T)
    db = np.sum(dZ)
    return dW, db

def train_logical_unadaptive(X, Y, num_iterations, learning_rate, plot_mid_train = False):
    w, b = initialize_with_zeros(len(X))
    costs = []
    if plot_mid_train:
        plt.ion()
        plt.show()
    for i in range(num_iterations):
        A, J = forward_propagation(X, Y, w, b)
        dW, db = backward_propagation(X, Y, A)
        w -= dW * learning_rate
        b -= db * learning_rate
        costs.append(J)
        if i%(num_iterations//50)==0:
            costs.append(J)
            print("Iteration: ", i, " cost: ", J)
            if plot_mid_train:
                plt.pause(0.0001)
                plt.clf()
                plt.plot(range(len(costs)), costs)
    if plot_mid_train:
        plt.pause(5)
    return w, b

def train_logical_adaptive(X, Y, num_iterations, learning_rate, plot_mid_train = False):
    w, b = initialize_with_zeros(len(X))
    costs = []
    learning_rate_W = np.full((len(w), 1), learning_rate)
    learning_rate_b = learning_rate
    if plot_mid_train:
        plt.ion()
        plt.show()
    for i in range(num_iterations):
        A, J = forward_propagation(X, Y, w, b)
        dW, db = backward_propagation(X, Y, A)
        learning_rate_W *= np.where(learning_rate_W * dW > 0, 1.1, -0.5)
        learning_rate_b *= 1.1 if db * learning_rate_b > 0 else -0.5
        w -= learning_rate_W
        b -= learning_rate_b
        costs.append(J)
        if i%(num_iterations//50)==0:
            print("Iteration: ", i, " cost: ", J)
            costs.append(J)
            if plot_mid_train:
                plt.pause(0.0001)
                plt.clf()
                plt.plot(range(len(costs)), costs)
    if plot_mid_train:
        plt.pause(10)
    return w, b

#linear
def calc_J_np_v2(X, Y, W, b):
    m, n = len(Y), len(W)
    dw = np.zeros((n, 1))
    J, db = 0, 0
    yHat = W.T@X+b
    diff = (yHat - Y)
    J = np.sum((diff**2)/m)
    dw = np.sum(2*X*diff, axis=1, keepdims=True)/m
    db = np.sum(2 * diff)/m
    return J, dw, db

def calc_J_np_v1(X, Y, W, b):
    m = len(X[0])
    n = len(W)
    dw = np.zeros((n, 1))
    J = 0
    db = 0
    for i in range(m):
        yHat = b
        Xi = X[:,i].reshape(len(W),1)
        yHat += Xi.T@W
        diff = (float)(yHat - Y[i])
        J += (diff**2)/m
        dw += (2*diff/m)*Xi
        db += 2 * diff/m
    return dw, db, J

def calcJ(X, Y, W, b):
    m = len(X[0])
    n = len(W)
    dw = [0] * n
    J = 0
    db = 0
    for i in range(m):
        yHat = b
        for j in range(n):
            yHat += W[j] * X[j][i]
        diff = (float)(yHat - Y[i])
        J += (diff**2)/m
        for j in range(n):
            dw[j] += 2*X[j][i]*diff/m
        db += 2 * diff/m
    return dw, db, J

def initFunc():
    xVals, yVals = b_utils.load_dataB1W4_trainN()
    weights = []
    
    b = rnd.randint(-100, 100)
    for i in xVals:
        weights.append(rnd.randint(-100, 100))
    return xVals, yVals, b, weights

def train_n_adaptive(X, Y, alpha, num_iterations, calc_J, plot_mid_train = False):
    if plot_mid_train:
        plt.ion()
        plt.show()
    m, n = len(Y), len(X)
    costs, b = [], 0
    W = np.zeros((n, 1))
    alpha_W = np.full((n, 1), alpha)
    alpha_b = alpha
    for i in range(1, num_iterations+1):
        cost, dW, db = calc_J(X, Y, W, b)
        alpha_W *= np.where(alpha_W * dW > 0, 1.1, -0.5)
        alpha_b *= 1.1 if db * alpha_b > 0 else -0.5
        W -= alpha_W
        b -= alpha_b
        if i%(num_iterations//50)==0:
            costs.append(cost)
            if plot_mid_train:
                plt.pause(0.0001)
                plt.clf()
                plt.plot(range(len(costs)), costs)
    return costs, W, b
    

def trainUnadaptive(iterations):
    xVals, yVals, b, weights = initFunc()
    learningRate = 0.1**13
    jVals = []
    for iter in range(iterations):
        dw, db, j = calcJ(xVals, yVals, weights, b)
        for w in range(len(weights)): 
            weights[w] -= learningRate * dw[w]  
        b -= learningRate * db
        if iter % 1000 == 0 and iter > 1000:
            print("weights:", [np.round(w, 2) for w in weights])
            print("b: ", b)
            jVals.append(j)
            plt.pause(0.1)
            plt.clf()
            plt.plot(range(len(jVals)), jVals)

def trainAdaptive(iterations):
    xVals, yVals, b, weights = initFunc()
    jVals = []
    learningRates = [0.1**-4] * (len(weights)+1)
    for iter in range(iterations):
        dw, db, j = calcJ(xVals, yVals, weights, b)
        for w in range(len(weights)): 
            learningRates[w] *= 1.1 if learningRates[w] * dw[w] > 0 else -0.5
            weights[w] -= learningRates[w] 
        learningRates[-1] *= 1.1 if learningRates[-1] * db > 0 else -0.5
        b -= learningRates[-1]
        if iter%(iterations/100) == 0 and iter > 0:
            print("weights:", [np.round(w, 2) for w in weights])
            print("b: ", b)
            print("j: ", j)
            jVals.append(j)
            plt.pause(0.1)
            plt.clf()
            plt.plot(range(len(jVals)), jVals)

def trainAdaptive(xVals, yVals, learningRate, iterations):
    arr, arr2, b, weights = initFunc()
    jVals = []
    learningRates = [learningRate] * (len(weights)+1)
    for iter in range(iterations):
        dw, db, j = calcJ(xVals, yVals, weights, b)
        for w in range(len(weights)): 
            learningRates[w] *= 1.1 if learningRates[w] * dw[w] > 0 else -0.5
            weights[w] -= learningRates[w] 
        learningRates[-1] *= 1.1 if learningRates[-1] * db > 0 else -0.5
        b -= learningRates[-1]
        if iter%(iterations/100) == 0 and iter > 0:
            print("weights:", [np.round(w, 2) for w in weights])
            print("b: ", b)
            print("j: ", j)
            jVals.append(j)
            plt.pause(0.1)
            plt.clf()
            plt.plot(range(len(jVals)), jVals)
    return jVals, weights, b



main()