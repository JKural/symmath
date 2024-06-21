import numpy as np
import matplotlib.pyplot as plt

import symmath as sm

d = sm.Derivative(0)

f = 1/np.sqrt(2*np.pi)*sm.exp(-sm.var(0)**2/2)
print("f = 1/np.sqrt(2*np.pi)*sm.exp(-sm.var(0)**2/2)")
print("f: ", f)

xs = np.linspace(-4, 4, 1000)
for i in range(4):
    plt.plot(xs, f(xs), label=f"d^{i}(f)")
    f = d(f)
plt.legend()
plt.suptitle("Pochodne gęstości rozkładu normalnego")
plt.show()