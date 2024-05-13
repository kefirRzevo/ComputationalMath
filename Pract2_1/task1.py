import numpy as np
from numpy import float32, float64
import matplotlib.pyplot as plt
import os

# https://perso.crans.org/besson/publis/notebooks/Runge-Kutta_methods_for_ODE_integration_in_Python.html
def rk4(fs: np.array, xs: np.array, t: float, step: float, **kwargs):
    def get_next_value(f, y):
        k_1 = f(xs, t, **kwargs)
        k_2 = f(xs + k_1 * step / 2.0, t + step / 2.0, **kwargs)
        k_3 = f(xs + k_2 * step / 2.0, t + step / 2.0, **kwargs)
        k_4 = f(xs + k_3 * step, t + step, **kwargs)
        return y + (step / 6.0) * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)

    return np.vectorize(get_next_value)(fs, xs)

def calculate_tension(xs: np.array, t, m: float, f, l, *args, **kwargs) -> float:
    return (m * (xs[1] ** 2.0 + xs[3] ** 2.0) - xs[2] * f(t, m)) / l

def f_1(xs: np.array, *args, **kwargs) -> float:
    return xs[1]

def f_2(xs: np.array, t, m, l, f, *args, **kwargs) -> float:
    tension = calculate_tension(xs, t, m, f, l)
    return -1.0 * xs[0] / (m * l) * tension

def f_3(xs: np.array, *args, **kwargs) -> float:
    return xs[3]

def f_4(xs: np.array, t, m, l, f, *args, **kwargs) -> float:
    tension = calculate_tension(xs, t, m, f, l)
    return -1.0 * xs[2] / (m * l) * tension - f(t, m) / m

def j_1(xs: np.array, *args, **kwargs) -> np.array:
    return np.array([0.0, 1.0, 0.0, 0.0])

# https://www.wolframalpha.com/input?i2d=true&i=Partial%5BDivide%5B-Subscript%5Bx%2C1%5D%2Cm+l%5DDivide%5Bm%2Cl%5D%5C%2840%29Power%5BSubscript%5Bx%2C2%5D%2C2%5D%2BPower%5BSubscript%5Bx%2C4%5D%2C2%5D%5C%2841%29+%2B+Subscript%5Bx%2C3%5DDivide%5BSubscript%5Bx%2C1%5D%2Cm+l%5D+f%5C%2840%29t%5C%2841%29%2CSubscript%5Bx%2C1%5D%5D
def j_2(xs: np.array, t, m, l, f, *args, **kwargs) -> np.array:
    return np.array(
        [
            (l * xs[2] * f(t, m) - m * xs[2] ** 2.0) / (l**2.0 * m)
            - xs[3] ** 2.0 / l**2.0,
            -2.0 * xs[0] * xs[1] / l**2.0,
            xs[0] * f(t, m) / (l * m),
            -2.0 * xs[0] * xs[3] / l**2.0,
        ]
    )

def j_3(xs: np.array, *args, **kwargs) -> np.array:
    return np.array([0.0, 0.0, 0.0, 1.0])

def j_4(xs: np.array, t, m, l, *args, **kwargs) -> np.array:
    return np.array(
        [
            0.0 - 2.0 * xs[1] * xs[2] / l**2.0,
            (2 * l * xs[2] * f(t, m) - m * (xs[1] ** 2.0 + xs[3] ** 2.0))
            / (l**2.0 * m),
            -2.0 * xs[2] * xs[3] / l**2.0,
        ]
    )

m = 2.0  # kg
l = 8.0  # m
g = lambda t: 2 * np.cos(2 * np.pi * t)  # additional force
f = lambda t, m: m * (9.81 + g(t))  # gravity
fs = np.array([f_1, f_2, f_3, f_4])  # right-hand side
js = np.array([j_1, j_2, j_3, j_4])  # jacobians for right-hand side
xs_0 = np.array([3.0, 0.0, -np.sqrt(55.0), 0.0])  # initial position
step = 0.001  # seconds

t = 0.0  # initial time
t_b = 8.0  # upper bound on time
xs = xs_0  # initial conditions
solution = []
ts = []

while t <= t_b:
    xs = rk4(fs, xs, t, step, m=m, l=l, f=f)
    solution.append(xs)
    ts.append(t)
    t += step

t = np.array(list(map(lambda y: y, ts)))
x = np.array(list(map(lambda y: y[0], solution)))
y = np.array(list(map(lambda y: y[2], solution)))
z = []

for i in range(len(x)):
    z.append(l**2 - (x[i]**2+y[i]**2))


plt.figure(figsize = (8, 6))
#plt.plot(t, x, 'r', label = "x(t)")
#plt.plot(t, y, 'b', label = "y(t)")
plt.plot(t, z, 'b', label = "z(t)")
plt.legend()
plt.grid()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(11.7, 8.3))
ax.set_title("Рис. 1. Решение явной схемой RK4")
ax.plot(x, y)
rk4PathPdf = os.path.join(os.path.dirname(__file__), 'res/task1-explicit-rk4.pdf')
fig.savefig(rk4PathPdf, transparent=False, bbox_inches="tight")

def newton_solve(x0, f, jacob, eps: float64, norm=np.inf, max_iter=10000) -> np.array:
    cur = prev = x0

    for i in range(max_iter):
        j = jacob(*prev)
        inv_jacob = np.linalg.inv(j)
        delta = np.matmul(inv_jacob, f(*prev))
        cur -= delta
        err = np.linalg.norm(delta, norm)
        if err < eps:
            return cur

    raise RuntimeWarning("Simple iteration does not converge or is slowly converging")

# Implementation of Explicit and Implicit Runge-Kutta methods to
# reduce the computational cost of pollutant transport modeling
# Ioannis Charis
# https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_method
def gl4(
    fs: np.array,
    js: np.array,
    xs: np.array,
    t: float,
    step: float,
    eps: float,
    max_iter: int = 1000,
    **kwargs
):
    # Butcher table for Gauss–Legendre
    a11 = 0.25
    a12 = 0.25 - np.sqrt(3.0) / 6.0
    a21 = 0.25 + np.sqrt(3.0) / 6.0
    a22 = 0.25

    b1 = 0.5
    b2 = 0.5

    c1 = 0.5 - np.sqrt(3.0) / 6.0
    c2 = 0.5 + np.sqrt(3.0) / 6.0

    def get_next_value(f, j, y):
        k = f(xs, t, **kwargs)

        xs_1_guess = xs + c1 * step * k
        xs_2_guess = xs - c2 * step * k

        k_1_guess = f(xs_1_guess, t + c1 * step, **kwargs)
        k_2_guess = f(xs_2_guess, t + c2 * step, **kwargs)
        ks_0 = np.array([k_1_guess, k_2_guess])

        def f_for_k(k_1_cur, k_2_cur):
            return step * np.array(
                [
                    k_1_cur
                    - f(
                        xs + (k_1_cur * a11 + k_2_cur * a12) * step,
                        t + c1 * step,
                        **kwargs
                    ),
                    k_2_cur
                    - f(
                        xs + (k_1_cur * a21 + k_2_cur * a22) * step,
                        t + c2 * step,
                        **kwargs
                    ),
                ]
            )

        def j_for_k(k_1_cur, k_2_cur):
            j1 = j(xs + (k_1_cur * a11 + k_2_cur * a12) * step, t + c1 * step, **kwargs)
            j2 = j(xs + (k_1_cur * a12 + k_2_cur * a22) * step, t + c2 * step, **kwargs)

            return np.eye(2) - step * np.array(
                [
                    [a11 * np.sum(j1), a12 * np.sum(j1)],
                    [a21 * np.sum(j2), a22 * np.sum(j2)],
                ]
            )

        ks = newton_solve(ks_0, f_for_k, j_for_k, eps=eps, max_iter=max_iter)
        k_1, k_2 = ks
        return y + step * (b1 * k_1 + b2 * k_2)

    return np.vectorize(get_next_value)(fs, js, xs)

step = 0.01  # seconds
t = 0.0    # initial time
t_b = 4.0  # upper bound on time
xs = xs_0  # initial conditions
solution = []

while t <= t_b:
    xs = gl4(fs, js, xs, t, step, 1e-4, 1000, m=m, l=l, f=f)
    solution.append(xs)
    t += step

x = np.array(list(map(lambda y: y[0], solution)))
y = np.array(list(map(lambda y: y[2], solution)))

fig, ax = plt.subplots(1, 1, figsize=(11.7, 8.3))
ax.set_title("Рис. 2. Решение неявной схемой GL4")
ax.plot(x, y)
gl4PathPdf = os.path.join(os.path.dirname(__file__), 'res/task1-implicit-gl4.pdf')
fig.savefig(gl4PathPdf, transparent=False, bbox_inches="tight")
