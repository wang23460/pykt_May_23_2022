import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt

orig_x = np.array([5, 15, 25, 35, 45, 55])
print(orig_x.shape)
x = orig_x.reshape((-1, 1))
print(x.shape)
y = np.array([15, 11, 2, 8, 25, 32])
plt.plot(x, y)
plt.scatter(x, y)
plt.figure()

regression1 = LinearRegression()
regression1.fit(x, y)
x_sequence = np.array(np.arange(5, 55, 0.1)).reshape((-1, 1))
plt.plot(x, y)
plt.scatter(x, y)
plt.plot(x, regression1.coef_ * x + regression1.intercept_)
score1 = regression1.score(x, y)

plt.figure()
print(f"1st order regression = {score1}")

transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
print(f"x_ shape ={x_.shape}, x shape={x.shape} ")

regression2 = LinearRegression().fit(x_, y)
score2 = regression2.score(x_, y)
print(f"2nd order regression = {score2}")
print(f"2nd coef = {regression2.coef_}")
print(f"2nd intercept={regression2.intercept_}")
x_sequence_ = transformer.transform(x_sequence)
y_pred = regression2.predict(x_sequence_)
plt.plot(x, y)
plt.scatter(x, y)
plt.plot(x_sequence, y_pred)

plt.show()