import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/sonar.all-data', header=None, prefix='X')
print(df.shape)
print(df.head(5))
data, labels = df.iloc[:, :-1], df.iloc[:, -1]
print(data.shape)
print(labels.shape)
df.rename(columns={'X60': 'Label'}, inplace=True)
print(df.columns)

classifier = KNeighborsClassifier(n_neighbors=4)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
classifier.fit(X_train, y_train)
print("[train]score=", classifier.score(X_train, y_train))
print("[test]score=", classifier.score(X_test, y_test))