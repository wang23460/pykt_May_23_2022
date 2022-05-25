from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()
pca = PCA(n_components=2)
print(f"iris data shape={iris.data.shape}")
data = pca.fit(iris.data).transform(iris.data)
# print(data)
print(f"new data shape={data.shape}")
print(data[:5, ])


datamax = data.max(axis=0)
datamin = data.min(axis=0)
n = 2000
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
# kernel= [linear|rbf|poly]
# linear 0.96
# linear, c=0.1 0.95
# linear, c=100 0.973
# poly, c=0.1 0.88
# poly, c=1 0.946
# poly, c=100 0.96
svc = svm.SVC(kernel='poly', C=100)
svc.fit(data, iris.target)
vectors = svc.support_vectors_
print(f"accuracy={svc.score(data, iris.target)}")
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])

plt.contour(X, Y, Z.reshape(X.shape), colors='k')
plt.scatter(vectors[:, 0], vectors[:, 1], c='red', s=180)
for c, s in zip([0, 1, 2], ['o', '+', 'x']):
    d = data[iris.target == c]
    plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)
plt.show()