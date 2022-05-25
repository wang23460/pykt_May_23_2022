import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn import svm

np.random.seed(20220525)
X, y = make_blobs(n_samples=40, centers=2)
#linear, poly, rbf
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

YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = classifier.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, color='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'], colors='k')
ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()