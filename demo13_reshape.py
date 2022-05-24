import numpy as np

a = np.zeros((10, 2))
b = a.T
c = b.view()
print(a.shape, b.shape, c.shape)
d = np.reshape(b, (5, 4))
e = np.reshape(b, (20,))
f = np.reshape(b, (20, -1))
g = np.reshape(b, (10, -1))
h = np.reshape(b, (-1, 20))
print(d.shape, e.shape, f.shape, g.shape, h.shape)
print(e)
print(f)
print(h)