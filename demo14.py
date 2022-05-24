import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a.view() # b shape改變不影響a, a內容改變影響b, a shape改變不影響b
c = a
print(a)
print(b)
print(c)
b.shape = (4, -1)
print(a)
print(b)
print(c)
c.shape = (1, 4)
print("_____________________")
print(a)
print(b)
print(c)
print("***************************")
a[0][0] = 100
print(a)
print(b)
print(c)