import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(type(diabetes))
features = diabetes.data
targets = diabetes.target
print(features.shape, targets.shape)

dataForTest = -60
data_train = features[:dataForTest]
target_train = targets[:dataForTest]
data_test = features[dataForTest:]
target_test = targets[dataForTest:]

regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)
print(regression1.coef_)
print(regression1.intercept_)


print(f'score:{regression1.score(data_test, target_test)}')
for i in range(dataForTest, 0):
    d = np.array(data_test[i]).reshape(1, -1)
    print(f"predict={regression1.predict(d)[0]:.2f}, actual={target_test[i]:.2f}")
mean_square_error = np.mean((regression1.predict(data_test) - target_test) ** 2)
print(f"MSE(mean square error)={mean_square_error},RMSE={mean_square_error ** 0.5}")