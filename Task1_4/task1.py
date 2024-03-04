import numpy as np
import math
import matplotlib.pyplot as plt

#first function x^2 + y^2 = 1
def f1(X):
    return math.sqrt(1 - X**2)

#second function y = tg x
def f2(X):
    return math.tan(X)

#iteration scheme
def getNextX(X):
    return math.atan(math.sqrt((1 - X**2)))

#takes start point and epsilon
def solve(X0, eps):
    Xk = X0
    Xk1 = getNextX(Xk)
    while (math.fabs(Xk1 - Xk) > eps):
        Xk = Xk1
        Xk1 = getNextX(Xk)
    return Xk1

def main():
    plt.figure(figsize = (6, 6))

    X = np.arange(-1, 1, 0.001)
    Y = [f1(i) for i in X]
    plt.plot(X, Y, 'r')

    Y = [-i for i in Y]
    plt.plot(X, Y, 'r')

    Y = [f2(i) for i in X]
    plt.plot(X, Y, 'b')

    SolutionX = [0.65, -0.65]
    SolutionY = [0.76, -0.76]
    plt.scatter(SolutionX, SolutionY, marker = "^")

    plt.grid()
    plt.show()

    eps = 1e-6
    X0 = 0.6
    X = solve(X0, eps)
    print("Nonlinear equations system solution is: x = +-", format(X, '.6f'), ", y = +-", format(math.tan(X),  '.6f'))

if __name__ == '__main__':
    main()
