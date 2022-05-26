from matplotlib import pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8], [3, 9]]
values = [1, 8, 5.5, 9]
regression1 = linear_model.LinearRegression()
regression1.fit(features, values)
print(f"coef_={regression1.coef_}")
print(f"intercept={regression1.intercept_}")
print(f"score={regression1.score(features, values)}")
# score means r-square


# features = [[0, 1], [1, 3], [2, 8]]
# values = [1, 8, 5.5]
# regression1 = linear_model.LinearRegression()
# regression1.fit(features, values)
# print(f"coef_={regression1.coef_}")
# print(f"intercept={regression1.intercept_}")
# print(f"score={regression1.score(features, values)}")