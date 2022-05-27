import numpy
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

dataset = numpy.loadtxt("data/diabetes.csv", delimiter=",", skiprows=1)

inputList = dataset[:, :8]
resultList = dataset[:, 8]
print(inputList.shape, resultList.shape)


def createModel():
    # global model
    m = Sequential()
    layers = [Dense(10, input_dim=8, activation='relu'),
              Dense(8, activation='relu'),
              Dense(1, activation='sigmoid')]
    for l in layers:
        m.add(l)
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    m.summary()
    return m


model = KerasClassifier(build_fn=createModel, epochs=200, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
result = cross_val_score(model, inputList, resultList, cv=fiveFold)
print(f"mean={result.mean()}, std={result.std()}")