import numpy as np
from DL2 import *

def check_grad(f, x, f_grad, epsilon=1e-7):
    approx = (f(x+epsilon)-f(x-epsilon))/(2*epsilon)
    diff = abs(f_grad(x)-approx)/(abs(f_grad(x))+abs(approx))
    check = (diff < epsilon)
    return check, diff

def check_n_grad(f, params_vec, grad_vec, epsilon=1e-7):
    n = len(params_vec)
    approx = np.zeros((n,), dtype=float)
    for i in range(n):
        v = np.copy(params_vec)
        v[i] += epsilon
        f_plus = f(v)
        v[i] -= 2*epsilon 
        f_minus = f(v) 
        approx[i] = f_plus-f_minus
    approx /= (2*epsilon) 
    diff = (np.linalg.norm(grad_vec-approx))/((np.linalg.norm(grad_vec))+(np.linalg.norm(approx)))
    check = (diff < epsilon)
    return check, diff

def g(parms):
    a,b = parms[0], parms[1]
    return 2*a**2+4*a*b-3*b**2

def dg_da(a,b):

    return 4*a+4*b

def dg_db(a,b):

    return 4*a-6*b

def dg_db_wrong(a,b):

    return 4*a-6*b+0.001

np.random.seed(3)

check_X = np.random.randn(3,1)

check_Y = np.sum(check_X,axis=0) > 2

hidden1 = DLLayer("h1",2,(3,),"relu",W_initialization = "Xavier")

hidden2 = DLLayer("h2",3,(2,),"relu",W_initialization = "Xavier")

hidden3 = DLLayer("h3",10,(3,),"sigmoid",W_initialization = "Xavier")

model = DLModel()

model.add(hidden1)

model.add(hidden2)

model.add(hidden3)

model.compile("squared_means")

check, diff, layer = model.check_backward_propagation(check_X, check_Y)

print("check:",str(check), ", diff:", str(diff), ", layer:", str(layer))