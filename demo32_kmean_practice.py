from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np

X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]
for e in X:
    plt.scatter(e[0], e[1], c='black', s=7)

K = 3
C_x = np.random.uniform(np.min(X[:, 0]), np.max(X[:, 0]), size=K)
C_y = np.random.uniform(np.min(X[:, 1]), np.max(X[:, 1]), size=K)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)
plt.scatter(C_x, C_y, marker='*', s=100, c='#0599FF')
plt.show()

#手做 K_MEANS
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
delta = dist(C, C_old, None)
print(f"delta={delta}")

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']


def plot_kmean(current_cluster, delta):
    fig, ax = plt.subplots()
    for index1 in range(K):
        points = np.array([X[j] for j in range(len(X)) if current_cluster[j] == index1])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[index1])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='#0599FF')
    plt.title(f'delta is:{delta:.2f}')
    plt.show()


while delta != 0:
    # do something
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    for i in range(K):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    delta = dist(C, C_old, None)
    print(f"delta={delta}")
    plot_kmean(clusters, delta)