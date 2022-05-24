from matplotlib import pyplot as plt
from sklearn import datasets

X, y = datasets.make_regression(10, 6, noise=5)
print(X)
print('---')
print(y)
for i in range(6):
    plt.figure()
    x1 = X[:, i]
    plt.scatter(x1, y)

plt.show()