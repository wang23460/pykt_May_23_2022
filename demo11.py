from sklearn import datasets

data = datasets.make_regression(10, 6, noise=5)

X = data[0]
print(type(X), X.shape)
print(X)

r1 = sorted(X, key=lambda r: r[0])
r2 = sorted(X, key=lambda r: r[1])
r3 = sorted(X, key=lambda r: r[2])
r4 = sorted(X, key=lambda r: r[3])
r5 = sorted(X, key=lambda r: r[4])
r6 = sorted(X, key=lambda r: r[5])

print(r1)
print("program finished")