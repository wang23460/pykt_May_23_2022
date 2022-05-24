from matplotlib import pyplot as plt
from sklearn import linear_model

regression1 = linear_model.LinearRegression()
# features = [[1], [2], [3]]
features = [[1], [2], [3]]
values = [1, 4, 15]
plt.scatter(features, values, c='green')
#plt.show()
plt.figure()
regression1.fit(features, values)
print("coef", regression1.coef_)
print("intercept", regression1.intercept_)
range1 = [0, 4]
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, c='gray')
plt.scatter(features, values, c='green')
plt.show()