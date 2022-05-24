
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import numpy

data1 = datasets.make_regression(100, 1, noise=50)
print(type(data1), len(data1))
print(type(data1[0]), type(data1[1]))
print(data1[0].shape, data1[1].shape)
plt.scatter(data1[0], data1[1], c='red', marker='.')
plt.figure()

regression1 = linear_model.LinearRegression()
regression1.fit(data1[0], data1[1])
print(f"coef={regression1.coef_}, intercept={regression1.intercept_}")
print(f"score={regression1.score(data1[0], data1[1])}")

range1 = numpy.arange(data1[0].min() - 0.1, data1[0].max() + 0.1, 0.1)

plt.plot(range1, regression1.coef_*range1+regression1.intercept_)
plt.scatter(data1[0], data1[1], c='red', marker='.')
plt.show()