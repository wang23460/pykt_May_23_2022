from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]  # setosa, versicolor, verginica

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]
classifier1 = SVC(kernel="linear", C=float(99999))
classifier1.fit(X, y)
print(classifier1.coef_)
print(classifier1.intercept_)

# y = ax+b
x0 = np.linspace(0, 5.5, 200)
pred1 = np.random.randint(1, 5) * x0 - np.random.randint(1, 5)
pred2 = x0 - np.random.randint(1, 5)
pred3 = np.random.randn(1) * x0 + np.random.randn(1)


def plot_svc_linear_boundary(classifier, xmin, xmax):
    w = classifier.coef_[0]
    b = classifier.intercept_[0]
    x = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1] # SVC General Form
    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    vectors = classifier.support_vectors_
    plt.scatter(vectors[:, 0], vectors[:, 1], s=180, facecolor="#C0FFEE")
    plt.plot(x, decision_boundary, "k-", linewidth=3, color='green')
    plt.plot(x, gutter_up, "k--", linewidth=2)
    plt.plot(x, gutter_down, "k--", linewidth=2)


fig, axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey="all")
# random guess
plt.sca(axes[0])
plt.plot(x0, pred1, 'g--', linewidth=2)
plt.plot(x0, pred2, 'm--', linewidth=2)
plt.plot(x0, pred3, 'r--', linewidth=2)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="versicolor")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="setosa")
# by svc
plt.sca(axes[1])
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="versicolor")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="setosa")
plot_svc_linear_boundary(classifier1, 0, 5.5)
plt.show()