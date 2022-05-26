from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
data = iris.data
target = iris.target

classifiers = [LogisticRegression(), SVC(kernel="linear"),
               SVC(kernel='poly'), SVC(kernel='rbf'), DecisionTreeClassifier(),
               KNeighborsClassifier(n_neighbors=2), KNeighborsClassifier(n_neighbors=4),
               KNeighborsClassifier(n_neighbors=6),
               GaussianNB(), RandomForestClassifier(n_estimators=100)]
for clf in classifiers:
    print(f"classifier={clf}")
    score = model_selection.cross_val_score(clf, data, target, cv=3)
    print(score)