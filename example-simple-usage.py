import symmath as sm

# f = 3
f = sm.const(3)
print("f = sm.const(3)")
print("repr(f): ", repr(f))
print("str(f):  ", f)
print()

# f(x) = 2**x
g = 2**sm.var(0) + sm.var(0)**2
print("g = 2**sm.var(0) + sm.var(0)**2")
print("repr(g): ", repr(g))
print("str(g):  ", g)
print()

# h(x, y) = x*0 + y**0
h = sm.var(0) * 0 + sm.var(1)**0
print("h = sm.var(0) * 0 + sm.var(1)**0")
print("repr(h): ", repr(h))
print("str(h):  ", h)
print("reduce(h): ", sm.reduce(h))
print()
