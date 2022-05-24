from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
from sklearn.linear_model import LinearRegression
import numpy
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

X, y = datasets.make_regression(1000, 10, n_informative=5, noise=25)
model = LinearRegression()
model.fit(X, y)
importance = model.coef_
print(f"importance={importance}")

for i, v in enumerate(importance):
    print(f"feature{i} score:{v}")
plt.bar([x for x in range(len(importance))], importance)

kBest = SelectKBest(f_regression, k=5).fit(X, y)
print(kBest.get_support())
newX = kBest.fit_transform(X, y)
print(X[:1])
print(newX[:1])
plt.show()