import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn import svm

X, y = make_blobs(n_samples=40, centers=2)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
classifier = svm.SVC(kernel='linear', C=float("inf"))
classifier.fit(X, y)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
print(xlim)
print(ylim)
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()