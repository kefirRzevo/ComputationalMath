import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sp

def P(x):
    return x*x-3

def Q(x):
    return math.cos(x)*(x*x-3)

def F(x):
    return 2-6*x+2*x*x*x+(x*x-3)*math.exp(x)*math.sin(x)*(1+math.cos(x))+math.cos(x)*(math.exp(x)+(x*x-1)+x*x*x*x-3*x*x)

def A():
    return 0

def B():
    return math.pi

def Ya():
    return 0

def Yb():
    return math.pi*math.pi

def Xn(a, h, N):
    X = np.zeros(N+1)
    for i in range(N+1):
        X[i] = a + i*h
    return X

#y"+py'+y=f
def cauchy2order(p, q, f, yAtStart, yDerAtStart, X, h, N):
    P = np.zeros(N+1)
    Q = np.zeros(N+1)
    F = np.zeros(N+1)
    Y = np.zeros(N+1)
    for i in range(N+1):
        P[i] = p(X[i])
        Q[i] = q(X[i])
        F[i] = f(X[i])
    Y[0] = yAtStart
    Y[1] = yAtStart + h * yDerAtStart
    for i in range(0, N - 1):
        Y[i+2] = (F[i]-Q[i]*Y[i])*h*h-P[i]*(Y[i+1]-Y[i])*h+2*Y[i+1]-Y[i]
    return Y

#Z1'=-qZ1^2-pZ1+1
def cauchy1orderZ1(p, q, z1AtStart, X, h, N):
    P = np.zeros(N+1)
    Q = np.zeros(N+1)
    Z1 = np.zeros(N+1)
    for i in range(N+1):
        P[i] = p(X[i])
        Q[i] = q(X[i])
    Z1[0] = z1AtStart
    for i in range(0, N):
        Z1[i+1] = (Z1[i]*Z1[i]*Q[i]+Z1[i]*P[i]+1)*h+Z1[i]
    return Z1

#Z2'=-Z1(Z2q+f)
def cauchy1orderZ2(q, f, Z1, yAtStart, X, h, N):
    F = np.zeros(N+1)
    Q = np.zeros(N+1)
    Z2 = np.zeros(N+1)
    for i in range(N+1):
        Q[i] = q(X[i])
        F[i] = f(X[i])
    Z2[0] = yAtStart
    for i in range(0, N):
        Z2[i+1] = (Z1[i]*Z2[i]*Q[i]-Z1[i]*F[i])*h+Z2[i]
        #print(Z2[i])
    return Z2

#Z1y'=y-Z2
def cauchy1orderY(Z1, Z2, yAtEnd, h, N):
    Y = np.zeros((N+1))
    Y[N] = yAtEnd
    print(Y[N])
    for i in range(N, 0, -1):
        Y[i-1] = (Y[i]*Z1[i-1]+Z2[i-1]*h)/(Z1[i-1]+h)
    return Y

def commonSolution(a, b, ya, yb, p, q, f, N=100):
    h = (b - a) / N
    X = Xn(a, h, N)
    def zero(x):
        return 0
    YPart = cauchy2order(p, q, f, 0, 0, X, h, N)
    Y1 = cauchy2order(p, q, zero, 1, 0, X, h, N)
    Y2 = cauchy2order(p, q, zero, 0, 1, X, h, N)
    Y = YPart + ya * Y1 + (yb - YPart[N] - ya*Y1[N])/Y2[N]*Y2
    return X, Y

def runningSolution(a, b, ya, yb, p, q, f, N=100):
    h = (b - a) / N
    X = Xn(a, h, N)
    Z1 = cauchy1orderZ1(p, q, 0, X, h, N)
    Z2 = cauchy1orderZ2(q, f, Z1, ya, X, h, N)
    Y = cauchy1orderY(Z1, Z2, yb, h, N)
    return X, Y

def main():
    X1, Y1 = commonSolution(A(), B(), Ya(), Yb(), P, Q, F)
    X2, Y2 = runningSolution(A(), B(), Ya(), Yb(), P, Q, F)
    plt.figure(figsize = (8, 6))
    plt.title("Solving the problem using the common solution and running solution:")
    plt.plot(X1, Y1, 'r', label = "common solution")
    plt.plot(X2, Y2, 'g', label = "running solution")
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    main()
