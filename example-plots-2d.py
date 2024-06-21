import numpy as np
import matplotlib.pyplot as plt

import symmath as sm

d0 = sm.Derivative(0)
d1 = sm.Derivative(1)

f = sm.cos(0.3*sm.var(0)+0.5*sm.var(1)) + sm.sin(0.8*sm.var(0)-0.2*sm.var(1))

xs = np.linspace(-10, 10, 1000)
ys = np.linspace(-10, 10, 1000)
xs, ys = np.meshgrid(xs, ys)

plt.contourf(xs, ys, f(xs, ys), levels=100, cmap=plt.cm.coolwarm)

xs = np.linspace(-10, 10, 20)
ys = np.linspace(-10, 10, 20)
xs, ys = np.meshgrid(xs, ys)

plt.quiver(xs, ys, d0(f)(xs, ys), d1(f)(xs, ys))
plt.show()