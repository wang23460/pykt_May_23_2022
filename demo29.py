from subprocess import check_call
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 0, 1, 1]
col = ['red', 'green']
marker = ['o', 'd']
index = 0

while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0], X[index][1], c=col[type], marker=marker[type])
    index += 1
plt.show()

classifier1 = tree.DecisionTreeClassifier()
classifier1.fit(X, Y)
print(classifier1.tree_)
# 建一個graph的目錄
export_graphviz(classifier1, out_file="graph/demo29.dot",
                filled=True, rounded=True,
                special_characters=True)
check_call(['dot','-Tpng','graph/demo29.dot','-o', 'graph/demo29.png'])
check_call(['dot', '-Tpdf', 'graph/demo29.dot', '-o', 'graph/demo29.pdf'])