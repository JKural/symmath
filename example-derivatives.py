import symmath as sm

d0 = sm.Derivative(0)
d1 = sm.Derivative(1)

# f(x, y) = x**2*y + 3
f = sm.var(0)**2 * sm.var(1) + 3
print("f = sm.var(0) + sm.var(1) + 3")
print("f: ", f)
print("d0(f): ", d0(f))
print("d1(f): ", d1(f))
print("d0(d1(f)): ", d0(d1(f)))
print("d1(d0(f)): ", d1(d0(f)))

# g(x) = sin(e^x)
g = sm.sin(sm.exp(sm.var(0)))
print("g = sm.var(0) * sm.exp(sm.var(0))")
for i in range(5):
    print(f"d^{i}g: ", g)
    g = d0(g)