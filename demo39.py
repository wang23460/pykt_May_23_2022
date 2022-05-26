import seaborn as sns
from sklearn import datasets
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = np.array([iris.target_names[i] for i in iris.target])
sns.pairplot(df, hue='species')
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target,
                                                    test_size=0.5, stratify=iris.target)
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
predicted = rf.predict(X_test)
print("rf out of box score:{:.3}".format(rf.oob_score_))

plt.show()
cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns=iris.target_names, index=iris.target_names)

print(cm)
sns.heatmap(cm, annot=True)
plt.show()