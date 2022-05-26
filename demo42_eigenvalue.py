from numpy.linalg import eig
from numpy import array, cov, mean

A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)
M = mean(A.T, axis=1)
print(M)
M2 = mean(A.T)
print(M2)
M3 = mean(A, axis=1)
print(M3)
C = A - M
print(C)
V = cov(C.T)
print(V)
values, vectors = eig(V)
print("values=", values)
print("vectors=", vectors)
p = vectors.T.dot(C.T)
print("project result=\n", p)