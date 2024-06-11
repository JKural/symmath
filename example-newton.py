import symmath as sm

def newton_method(f, x0, eps=1e-10, max_iter=1000):
    df = sm.Derivative(0)(f)
    for _ in range(max_iter):
        if abs(x0) < eps:
            break
        x0 = x0 - f(x0)/df(x0)
    return x0

f = sm.var(0)**2 - sm.var(0) - 1
print("f = sm.var(0)**2 - sm.var(0) - 1")
print("newton_method(f, 1): ", newton_method(f, 1))
print("newton_method(f, -1): ", newton_method(f, -1))