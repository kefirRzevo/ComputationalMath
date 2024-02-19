import numpy as np
import matplotlib.pyplot as plt
import math

def f(t, x):
    u = x[0]
    return np.array([x[1], -math.sin(u)])

def main():
    x = [[1, 0]] #[[u0, y0] , [u1, y1], ...]
    h = 10**-3
    l = 0
    r = 4 * math.pi
    A = np.array([[0.5 - math.sqrt(3) / 6, 0], [math.sqrt(3) / 3, -0.5 - math.sqrt(3) / 2]])
    B = np.array([1 + math.sqrt(3) / 6, -math.sqrt(3) / 6])
    C = np.array([0.5 - math.sqrt(3) / 6, -0.5 - math.sqrt(3) / 6])
    t = np.arange(l, r, h)
    n = len(t)
    for i in range(n - 1):
        x.append(Runge(t[i], x[-1], h, f, A, B, C))
    y = [x[i][0] for i in range(n)]
    plt.figure()
    plt.plot(t, y)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.tight_layout()
    plt.grid()
    plt.show()

def Runge(x, u, h, f, a, b, c):
    n = len(u)
    s = len(b)
    k = MPI(f, x, c, h, u, a, n, s)
    res = []
    for i in range(n):
        res.append(u[i] + h * np.dot(k[i], b))
    return res

def MPI(f, x, c, h, u, a, n, s):
    eps = 10**-3
    kold = np.zeros((n, s))
    k = np.zeros((n, s))
    for i in range(s):
        U = [u[I] + h * np.dot(a[i], kold[I]) for I in range(n)]
        F = f(x + c[i] * h, U)
        for j in range(n):
            k[j][i] = F[j]
    while(max(abs(k[i][j] - kold[i][j]) for i in range(n) for j in range(s)) > eps):
        for i in range(n):
            for j in range(s):
                kold[i][j] = k[i][j]
        for i in range(s):
            U = [u[I] + h * np.dot(a[i], kold[I]) for I in range(n)]
            F = f(x + c[i] * h, U)
            for j in range(n):
                k[j][i] = F[j]
    return k

if __name__ == "__main__":
    main()
