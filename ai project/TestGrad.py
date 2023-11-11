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

a,b = 5.0,1.0

check, diff = check_n_grad(g, np.array([a,b]), np.array([dg_da(a,b),dg_db(a,b)]))

print("check:",str(check), ", diff:", str(diff))

check, diff = check_n_grad(g, np.array([a,b]), np.array([dg_da(a,b),dg_db_wrong(a,b)]))

print("check:",str(check), ", diff:", str(diff))


