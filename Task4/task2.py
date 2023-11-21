import numpy as np
import math
import matplotlib.pyplot as plt

#first function xy - x^2 = 1.03
def f1(X):
    return (1.03 + X**2) / X

#second function -2x^3 + y^2 = 1.98
def f2(X):
    return math.sqrt(1.98 + 2 * X**3)

#iteration scheme
def nextXY(X, Y):
    X1 = ((Y**2 - 1.98) / 2)**(1.0 / 3.0)
    Y1 = X + 1.03 / X
    return X1, Y1

#takes start point and epsilon
def solve(X0, Y0, eps):
    Xk, Yk   = X0, Y0
    Xk1, Yk1 = nextXY(Xk, Yk)
    Count = 0
    while (math.sqrt((Xk1 - Xk)**2 + (Yk1 - Yk)**2) > eps):
        Count += 1
        Xk, Yk   = Xk1, Yk1
        Xk1, Yk1 = nextXY(Xk, Yk)
    print("Estimating number of iterations:", Count)
    return Xk1, Yk1

def main():
    plt.figure(figsize = (5, 5))

    X = np.arange(-0.99, 2, 0.001)
    Y = [f2(i) for i in X]
    plt.plot(X, Y, 'y')

    Y = [-i for i in Y]
    plt.plot(X, Y, 'y')

    X1 = np.arange(-1.5, -0.1, 0.001)
    X2 = np.arange(0.1, 2, 0.001)
    Y = [f1(i) for i in X1]
    plt.plot(X1, Y, 'r')
    Y = [f1(i) for i in X2]
    plt.plot(X2, Y, 'r')

    plt.grid()
    plt.show()

    eps = 1e-4
    X0, Y0 = 1.0, 2.0
    X, Y = solve(X0, Y0, eps)
    print("Nonlinear equations system solution is: x = +-", format(X, '.6f'), ", y = +-", format(math.tan(X),  '.6f'))

if __name__ == '__main__':
    main()
