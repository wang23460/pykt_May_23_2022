from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

iris = datasets.load_iris()
print(type(iris))
print(list(iris.keys()))
print(iris["target_names"])
print(iris["feature_names"])
print(iris["data"].shape)
print(iris["target"].shape)
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)
print(y)
regression1 = LogisticRegression()
regression1.fit(X, y)
print(regression1.coef_)
print(regression1.intercept_)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = regression1.predict_proba(X_new)
# 1/(1+np.exp(-(ax+b)))
y_calculate = 1 / (1 + np.exp(-(regression1.coef_ * X_new + regression1.intercept_)))

plt.plot(X, y, "gs")
plt.plot(X_new, y_prob[:, 1], "g-", label="is verginica")
plt.plot(X_new, y_prob[:, 0], "b--", label="not verginica")
plt.plot(X_new, y_calculate, "r*", label="calculated")
plt.legend()
plt.show()
print(regression1.predict([[1.5], [2.0], [2.5]]))