import numpy as np
from matplotlib import pyplot as plt
from numpy.core import machar
from unit10 import b_utils
import random as rnd

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
            raise Exception("vectors must be the same size for dot.")
        if self.isCol or (not other.isCol):
            raise Exception("vectors need to be a row and a column (respectively) for dot.")
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

#test

#j and training
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