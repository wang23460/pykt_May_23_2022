from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
pca = PCA(2)
print(iris.data.shape)
data = pca.fit(iris.data).transform(iris.data)
print(data.shape)