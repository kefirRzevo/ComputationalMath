import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sp
from scipy. integrate import odeint

def Jacobian(f, x, dx = 1e-6):
    n = len(x)
    func = f(x)
    jac = np.zeros((n, n))
    for j in range(n):
        if x[j] != 0:
            Dxj = abs(x[j]) * dx
        else:
            Dxj = dx
        offset = []
        for k, xi in enumerate(x):
            if k != j:
                offset.append(xi)
            else:
                offset.append(xi + Dxj)
        jac[:, j] = (f(offset) - func) / Dxj
    return jac, func

def NewtonMethod(f, x, eps = 1.0e-2):
    max_iter = 1000
    a = np.array([[1, 2, 8], [3, 5, 0], [3, 2, 1]])
    b = np.array([1, 2, 6])
    j = np.linalg.solve(a, b)
    
    for i in np.arange(0, max_iter, 1):
        J, fn = Jacobian(f, x)
        if np.sqrt(np.dot(fn, fn) / x.size) < eps:
            return x
        dx = np.linalg.solve(J, np.array(fn.transpose()))
        x = x - dx

def Iteration(f, y0, tBEG, tEND, tau, alpha):
    def F(y_next):
        return y_next - tau * alpha * f(t[i], y_next) - y[i] - tau * (1. - alpha) * f(t[i], y[i])
    t = np.arange(tBEG, tEND, tau)
    y = np.ones((t.size, 3))
    y[0] = y0
    for i in np.arange(0, t.size - 1, 1):
        y_next = np.array([0, 0, 0])
        y_next = y[i] + tau * f(t[i], y[i])
        y[i + 1] = NewtonMethod(F, y_next)
    return t, y

def f(t, u):
    x = u[0]
    y = u[1]
    a = u[2]
    DerX = x * (1 - 0.5 * x - 2. * y / (7. * a * a)) 
    DerY = x * (2. * a - 3.5 * a * a * x - 0.5 * y)
    DerA = (2 - 7. * a * x) / 100
    return np.array([DerX, DerY, DerA])

def main():
    plt.figure(figsize = (8, 6))
    plt.title("Solving an ODE system using the 2nd order implicit Adams method:")
    tBEG = 0.
    tEND = 0.1
    tau = 0.001
    y0 = np.array([1.5, 10., 0.1])
    alpha = 0.5
    t, y = Iteration(f, y0, tBEG, tEND, tau, alpha)

    xMas = []
    yMas = []
    aMas = []
    for i in range(int(y.size / 3)):
        xMas.append(y[i][0])
        yMas.append(y[i][1])
        aMas.append(y[i][2])
    plt.plot(t, xMas, 'r', label = "x")
    plt.plot(t, yMas, 'b', label = "y")
    plt.plot(t, aMas, 'g', label = "a")

    plt.legend()
    plt.grid()

if __name__ == '__main__':
    main()

def Func(u : list, t : float)  -> list:
    a, x, y = u
    DerA = (2. - 7. * a * x) / 100.
    DerX = x * (1 - 0.5 * x - 2. * y / (7. * a * a))  
    DerY = x * (2. * a - 3.5 * a * a * x - 0.5 * y)    
    return [DerA, DerX, DerY]

def check():
    plt.figure(figsize = (8, 6))
    plt.title("Validation using scipy.integrate:")

    Start = 0
    Stop = 0.1
    DeltaT = 0.001

    Args = np.arange(Start, Stop, DeltaT)
    u0 = [0.1, 1.5, 10]
    t = np.linspace(0, 10, 41)
    w = odeint(Func, u0, Args)
    a = w[:, 0]
    x = w[:, 1]
    y = w[:, 2]

    plt.plot(Args, x, 'r', label = "ans_x")
    plt.plot(Args, y, 'b', label = "ans_y")
    plt.plot(Args, a, 'g', label = "ans_a")
    
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    check()
