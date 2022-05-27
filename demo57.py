import numpy
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import keras

dataset = numpy.loadtxt("data/diabetes.csv", delimiter=",", skiprows=1)

inputList = dataset[:, :8]
resultList = dataset[:, 8]

feature_train, feature_test, label_train, label_test = train_test_split(inputList, resultList, test_size=0.25,
                                                                        stratify=resultList)
for data in [resultList, label_train, label_test]:
    cls, counts = numpy.unique(data, return_counts=True)
    for cl, co in zip(cls, counts):
        print(f"{int(cl)} counts==>{co / sum(counts)}")


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


model = createModel()

model.fit(inputList, resultList, epochs=100, batch_size=20)

scores = model.evaluate(inputList, resultList)
metrics = model.metrics_names
for s, m in zip(scores, metrics):
    print(f"[model1]:{m} score={s}")
