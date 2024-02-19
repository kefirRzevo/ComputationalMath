import numpy as np
import matplotlib.pyplot as plt
import math

def f(t, u):
    y1 = u[0]
    y2 = u[1]
    return np.array([y1 - y1 * y2, -y2 + y1 * y2])

def main():
    h = 10**-4
    l = 0
    r = 10
    t = np.arange(l, r, h)
    n = len(t)
    u = Solve(t, n, h, f, [2, 2])
    plt.figure(figsize=(13.5, 6.3))
    plt.subplot(1, 2, 1)
    plt.plot(t, [u[i][0] for i in range(n)])
    plt.title("y1(t)")
    plt.tight_layout()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(t, [u[i][1] for i in range(n)])
    plt.title("y2(t)")
    plt.tight_layout()
    plt.grid()
    plt.show()

def Solve(t, n, h, f, u0):
    u = [u0]
    N = len(u0)
    for i in range(2):
        u.append([u[i][I] + h * f(t[i], u[i])[I] for I in range(N)])
    for i in range(n - 3):
        u.append(Next(u[-3], u[-2], u[-1], t[i], t[i + 1], t[i + 2], h, f, N))
    return u

def Next(u__, u_, u, t__, t_, t, h, F, N):
    f__ = F(t__, u__)
    f_ = F(t_, u_)
    f = F(t, u)
    return [u_[I] + h * (23 * f[I] / 12 - 16 * f_[I] / 12 + 5 * f__[I] / 12) for I in range(N)]

if __name__ == "__main__":
    main()
