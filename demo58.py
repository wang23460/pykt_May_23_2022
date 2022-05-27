import numpy
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold

dataset = numpy.loadtxt("data/diabetes.csv", delimiter=",", skiprows=1)

inputList = dataset[:, :8]
resultList = dataset[:, 8]
print(inputList.shape, resultList.shape)

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
totalScores = []


def createModel():
    # global model
    m = Sequential()
    layers = [Dense(10, input_dim=8, activation='relu'),
              Dense(8, activation='relu'),
              Dense(1, activation='sigmoid')]
    for l in layers:
        m.add(l)
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # m.summary()
    return m


for train, test in fiveFold.split(inputList, resultList):
    print("a new run")
    model = createModel()
    model.fit(inputList[train], resultList[train], epochs=100, batch_size=20, verbose=0)
    scores = model.evaluate(inputList[test], resultList[test], verbose=0)
    totalScores.append(scores[1] * 100)
print(f"total 5 result, mean={numpy.mean(totalScores)}, std{numpy.std(totalScores)}")