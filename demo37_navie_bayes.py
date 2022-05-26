import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
classifier1 = GaussianNB()
classifier1.fit(X, Y)
print(classifier1.predict([[0, 0], [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]))

classifier2 = GaussianNB()
classifier2.partial_fit(X, Y, np.unique(Y)) #須提供unique類別，可自己加類別 要用np.array()
print("partial fit stage1:")
print(classifier2.predict([[0, 0], [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]))
classifier2.partial_fit([[0.5, 0.5]], [2])
print("partial fit stage2:")
print(classifier2.predict([[0, 0], [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]))