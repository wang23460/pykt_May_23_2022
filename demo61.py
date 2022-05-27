from pandas import read_csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score

dataFrame = read_csv('data/iris.data', header=None)
print(dataFrame.shape)
dataset = dataFrame.values
print(type(dataFrame), type(dataset))
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
print(features.shape)
print(np.unique(labels, return_counts=True))
print(dataFrame.describe())

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(np.unique(encoded_Y))
dumm_y = np_utils.to_categorical(encoded_Y)
print(dumm_y[:10])
print(dumm_y[50:60])
print(dumm_y[100:110])


def baseline_model():
    m = Sequential()
    m.add(Dense(8, input_dim=4, activation='relu'))
    m.add(Dense(3, activation='softmax'))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


estimator = KerasClassifier(build_fn=baseline_model, epochs=300, batch_size=10)
kfold = KFold(n_splits=3, shuffle=True)
result = cross_val_score(estimator, features, dumm_y, cv=kfold)
print("accuracy:{}, std:{}".format(result.mean() * 100, result.std()))