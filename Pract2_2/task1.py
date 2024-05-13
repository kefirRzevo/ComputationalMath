import numpy as np
import os
import matplotlib.pyplot as plt
from celluloid import Camera
from IPython.display import HTML

M = 50
N = 100
X = 10
T = 0.02
h = 2 * X / M
tau = T / N
sigma = tau / h

r1 = 13
r2 = 1.3
p1 = 1000000
p2 = 100000
gamma = 1.4

dir = os.path.join(os.path.dirname(__file__), 'res')

def matrix_A(U):
  r = U[0]
  ur = U[1]
  er = U[2]

  u = ur / r
  e = er / r
  c = np.sqrt(gamma * (gamma - 1) * e)

  A = np.zeros(shape = (3, 3))
  A[0][0] = 0
  A[0][1] = 1
  A[0][2] = 0
  A[1][0] = -1 * u * u
  A[1][1] = 2 * u
  A[1][2] = gamma - 1
  A[2][0] = -1 * u * e * gamma
  A[2][1] = gamma * e
  A[2][2] = u
  return A

def matrix_omega(U):
  r = U[0]
  ur = U[1]
  er = U[2]

  u = ur / r
  e = er / r
  c = np.sqrt(gamma * (gamma - 1) * e)

  Omega = np.zeros(shape = (3, 3))
  Omega[0][0] = -1 * u * c
  Omega[0][1] = c
  Omega[0][2] = gamma - 1
  Omega[1][0] = -1 * c * c
  Omega[1][1] = 0
  Omega[1][2] = gamma - 1
  Omega[2][0] = u * c
  Omega[2][1] = -1 * c
  Omega[2][2] = gamma - 1
  return Omega

def matrix_L(U):
  r = U[0]
  ur = U[1]
  er = U[2]

  u = ur / r
  e = er / r
  c = np.sqrt(gamma * (gamma - 1) * e)

  L = np.zeros(shape = (3, 3))
  L[0][0] = u + c
  L[1][1] = u
  L[2][2] = u - c
  return L

def matrix_U_bound():
  U_bound = np.zeros(shape=(M, 3))
  for i in range(M):
    if i - M/2 <= 0:
      U_bound[i][0] = r1
      U_bound[i][2] = p1 / (gamma - 1)
    else:
      U_bound[i][0] = r2
      U_bound[i][2] = p2 / (gamma - 1)
    U_bound[i][1] = 0
  return U_bound

def getX():
  x_mas = np.zeros(shape=(M))
  for i in range(M):
    x_mas[i] = (i - M/2) * h
  return x_mas

def getR(u_n):
  r_mas = np.zeros(shape=(M))
  for i in range(M):
    r_mas[i] = u_n[i][0]
  return r_mas

def getU(u_n):
  r_mas = np.zeros(shape=(M))
  for i in range(M):
    r_mas[i] = u_n[i][1] / u_n[i][0]
  return r_mas

def getE(u_n):
  r_mas = np.zeros(shape=(M))
  for i in range(M):
    r_mas[i] = u_n[i][2] / u_n[i][0]
  return r_mas

def main():
  global tau, sigma
  images = []
  fig, ax = plt.subplots(figsize=(10, 6))
  camera = Camera(fig)
  u_prev = matrix_U_bound()
  u_next = np.zeros(shape=(M, 3))
  t_passed = 0
  n = 0
  while(t_passed < T):
    A0 = matrix_A(u_prev[0])
    u_next[0] = u_prev[0] - sigma * np.dot(A0, u_prev[1] - u_prev[0])
    u_next[0] = u_prev[0]
    for m in range(1, M-1):
      L = matrix_L(u_prev[m])
      while (tau * abs(L.max()) / h > 1):
        tau = tau / 2
        sigma = sigma / 2
        print(tau)
      omega = matrix_omega(u_prev[m])
      A = matrix_A(u_prev[m])
      omega_inv = np.linalg.inv(omega)
      lhs = np.dot(A, u_prev[m+1] - u_prev[m-1])
      rhs = np.dot(np.dot(np.dot(omega_inv, abs(L)), omega), u_prev[m+1] - 2 * u_prev[m] + u_prev[m-1])
      u_next[m] = u_prev[m] - (lhs - rhs) * sigma / 2
    u_next[M-1] = u_prev[M-1]
    AM = matrix_A(u_prev[M-1])
    u_next[M-1] = u_prev[M-1] - sigma * np.dot(AM, u_prev[M-1] - u_prev[M-2])
    ax.plot(getX(), getR(u_next), c='blue')
    #ax.plot(getX(), getU(u_next), c='blue')
    #ax.plot(getX(), getE(u_next), c='blue')
    camera.snap()
    u_prev = u_next.copy()
    t_passed += tau
    n = n + 1
  animation = camera.animate()
  HTML(animation.to_html5_video())
  plt.show()

if __name__ == "__main__":
    main()
