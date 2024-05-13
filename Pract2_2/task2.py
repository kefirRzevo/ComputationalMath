import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

a = 100
len = 0.5

M = 100
h = len / M
N = 50
tau = h**2 / (4*a)

dir = os.path.join(os.path.dirname(__file__), 'res')

def calculate_fir():
  U = np.zeros(shape=(N,M,M))
  # Boundary conditions
  U[:, 0, :] = 120
  U[:, -1, :] = 40
  U[:, :, 0] = 20
  U[:, :, -1] = 80

  for k in range(1, N):
    for i in range(1, M - 1):
      for j in range(1, M - 1):
        Uxx = (U[k-1,i-1,j] - 2 * U[k-1,i,j] + U[k-1,i+1,j]) / h**2
        Uyy = (U[k-1,i,j-1] - 2 * U[k-1,i,j] + U[k-1,i,j+1]) / h**2
        U[k,i,j] = tau * a * (Uxx + Uyy) + U[k-1,i,j]
  return U

def calculate_sec():
  U = np.zeros(shape=(N,M,M))
  # Boundary conditions
  for i in range (M):
    for j in range(M):
      if (i-M/2+1)**2 + (j-M/2+1)**2 > (M/2-1)**2:
        U[:, i, j] = 120

  for k in range(1, N):
    for i in range(1, M - 1):
      for j in range(1, M - 1):
        if (i-M/2+1)**2 + (j-M/2+1)**2 <= (M/2-1)**2:
          Uxx = (U[k-1,i-1,j] - 2 * U[k-1,i,j] + U[k-1,i+1,j]) / h**2
          Uyy = (U[k-1,i,j-1] - 2 * U[k-1,i,j] + U[k-1,i,j+1]) / h**2
          U[k,i,j] = tau * a * (Uxx + Uyy) + U[k-1,i,j]
  return U

def plotheatmap(u_n, n):
  # Clear the current plot figure
  plt.clf()
  plt.title(f"Temperature at t = {n*tau:.6f} unit time")
  plt.xlabel("x")
  plt.ylabel("y")

  # This is to plot u_n (u at time-step n)
  plt.pcolormesh(u_n, cmap=plt.cm.jet, vmin=0, vmax=120)
  plt.colorbar()

def main():
  U1 = calculate_fir()
  def animate1(k):
    plotheatmap(U1[k], k)
  anim1 = animation.FuncAnimation(plt.figure(), animate1, interval=1, frames=N, repeat=False)
  anim1.save(dir + "/fir.gif")

  U2 = calculate_sec()
  def animate2(k):
    plotheatmap(U2[k], k)
  anim2 = animation.FuncAnimation(plt.figure(), animate2, interval=1, frames=N, repeat=False)
  anim2.save(dir + "/sec.gif")
    
if __name__ == '__main__':
  main()
