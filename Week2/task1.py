import numpy
import matplotlib.pyplot as plt
 
def f(x):
    return x*x
 
x = numpy.arange(0,4,0.01)
y = f(x)
 
plt.figure(figsize=(10,5))
plt.plot(x, y, 'b')
plt.grid(axis = 'both')
plt.show()

from scipy.misc import derivative


d = derivative(f, 1.0, dx=1e-3)

print(d)
