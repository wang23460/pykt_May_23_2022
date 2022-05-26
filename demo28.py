from sklearn import tree
from matplotlib import pyplot as plt

X = [[0, 0], [1, 1]]
Y = [0, 1]

classifier = tree.DecisionTreeClassifier()
classifier.fit(X, Y)
print(classifier)

tree.plot_tree(classifier)
plt.show()