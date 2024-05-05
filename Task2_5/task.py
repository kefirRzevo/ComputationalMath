import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import animation

L = 2.
h = 0.05
lam0 = 0.5
sigma = 2.
T0 = 2.

x = np.arange(0, L, h)
U0 = np.zeros(x.size) + 10e-4

# lambda(T) = lambda_0 * T^G
def Lam(t):
    global sigma, lam0
    return lam0 * t ** sigma

# lambda_+/- = ( lambda_m + lambda_{m+1/m-1} ) / 2.
def LamFun(uPlus, uMinus): 
    LamPlus = Lam(uPlus)
    LamMinus = Lam(uMinus)
    return (LamPlus + LamMinus) / 2.

def Approx(u, m):
    LamPlus = LamFun(u[m + 1], u[m])
    LamMinus = LamFun(u[m], u[m - 1])
    return (LamPlus * (u[m + 1] - u[m]) - LamMinus * (u[m] - u[m - 1])) / h ** 2

def A(u, m, tau, gamma):
    return gamma * tau * LamFun(u[m], u[m - 1]) / h ** 2

def B(u, m, tau, gamma):
    LamPlus = LamFun(u[m], u[m + 1 ])
    LamMinus = LamFun(u[m], u[m - 1 ])
    return -gamma * tau * (LamPlus + LamMinus) / h ** 2 - 1.

def C(u, m, tau, gamma):
    return gamma * tau * LamFun(u[m], u[m + 1]) / h ** 2

def D(u, m, tau, gamma):
    return u[m] + tau * (1. - gamma) * Approx(u, m)

def Matrix(u, tau, gamma):
    Mat = np.zeros((u.size, u.size))
    Mat[ 0 ][ 0 ] = 1
    for m in np.arange(1, u.size-1, 1):
        Mat[m][m - 1] = A(u, m, tau, gamma)
        Mat[m][m] = B(u, m, tau, gamma)
        Mat[m][m + 1] = C(u, m, tau, gamma)
    Mat[-1][-1] = 1.
    return Mat

def vecD(u, tau, gamma):
    f = np.copy(u)
    for k in np.arange(1, u.size-1, 1):
        f[ k ] = -D(u, k, tau, gamma)
    return f

def SixPointsMth(u, T, tau, gamma):
    #Six-point method
    uNew = np.copy(u)
    M = int(T/tau)
    for i in np.arange(0, M, 1):
        t = i * tau
        # Boundary condition at x = 0
        uNew[0] = T0 * t ** (1. / sigma)
        # Boundary condition for x = L
        uNew[-1] = 0.0
        uNew = np.linalg.solve(Matrix(uNew, tau, gamma), vecD(uNew, tau, gamma))
    return uNew

#Explicit schema
def Explicit(u, T, tau, gamma):
    uNew = np.copy(u)
    uOld = np.copy(u)
    M = int(T / tau)
    for i in np.arange(0, M, 1):
        # Boundary condition at x = 0
        uNew[0] = T0 * (i * tau) ** (1. / sigma)
        # Boundary condition for x = L
        uNew[ -1 ] = 0.0
        for j in np.arange(1, u.size - 1, 1):
            uNew[j] = uOld[j] + tau * Approx(uOld, j)
        uOld = uNew
    return uNew

    # Calculation of auxiliary solutions for the three-stage method
    # (With a wave, with a stroke, with a star)
def getUNew(u, T, tau, gamma, Koeff):
    uNew = np.copy(u)
    M = int(T/tau)
    for i in np.arange(0, M, 1):
        t = i * tau
        # Boundary condition at x = 0
        uNew[0] = T0 * t ** (1. / sigma)
        # Boundary condition for x = L
        uNew[-1] = 0.0
        uNew = np.linalg.solve(Matrix(uNew, tau * Koeff, gamma), vecD(uNew, tau, gamma))
    return uNew

#Three-step method
def ThreeStepMth(u, T, tau, gamma):
    uNew = np.copy(u)
    uNewWave = getUNew(uNew, T, tau, gamma, 1.)
    uNewSh = getUNew(uNew, T, tau, gamma, 0.5)
    uNewSt = getUNew(uNewSh, T, tau, gamma, 0.5)
    for j in np.arange(0, u.size - 1, 1):
        uNew[j] = 2 * uNewWave[j] - uNewSt[j]
    return uNew

# Exact solution
def Exact(x, T):
    u = np.zeros(x.size)
    for m in np.arange(0, u.size, 1):
        if m * h > T:
            u[m] = 0.0
        else:
            u[m] = (sigma * (T - m * h) / lam0) ** (1. / sigma)
    return u

# Displays U(x) for all four methods for one point in time Time
def showTimeMoment( Time,  Tau,  Gamma ):
    u11 = Exact(U0, Time)
    u12 = Explicit(U0, Time, Tau,  Gamma)
    u13 = SixPointsMth(U0, Time, Tau, Gamma)
    u14 = ThreeStepMth(U0, Time, Tau, 1)
    
    plt.figure(figsize = (20, 8))
    plt.title("Time t = " + str(Time))
    plt.rc('font', **{'size' : 20})
    plt.grid()
    plt.xlabel( 'X' )
    plt.ylabel( 'U' )
    plt.plot( x, u11, 'r-x',
              x, u12, 'g-',
              x, u13, 'b--',
              x, u14, 'c-x')
    plt.legend(['Exact', 'Explicit', 'Six-point', 'Three-step'])
    plt.show()

def main():
    Tau = 0.0001
    Gamma = 0.5 + h ** 2 / (12. * lam0 * Tau)

    Times = [0.5, 1., 1.5]
    for T in Times:
        showTimeMoment(T,  Tau,  Gamma)

if __name__ == '__main__':
    main()
