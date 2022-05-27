import numpy
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

dataset = numpy.loadtxt("data/diabetes.csv", delimiter=",", skiprows=1)

inputList = dataset[:, :8]
resultList = dataset[:, 8]
print(inputList.shape, resultList.shape)


def createModel(optimizer='adam', init='uniform'):
    # global model
    m = Sequential()
    layers = [Dense(10, input_dim=8, kernel_initializer=init, activation='relu'),
              Dense(8, activation='relu'),
              Dense(1, activation='sigmoid')]
    for l in layers:
        m.add(l)
    m.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m.summary()
    return m


model = KerasClassifier(build_fn=createModel, verbose=0)