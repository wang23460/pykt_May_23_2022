from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

count = 5000
X = np.r_[np.random.randn(count, 2) + [5, 5],
          np.random.randn(count, 2) + [0, -5],
          np.random.randn(count, 2) + [-5, 5]]
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 10), inertias, 'b*--')
plt.show()