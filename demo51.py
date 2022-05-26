# use keras
import numpy
from keras.layers import Dense
from keras.models import Sequential

dataset = numpy.loadtxt("data/diabetes.csv", delimiter=",", skiprows=1)

inputList = dataset[:, :8]
resultList = dataset[:, 8]
print(inputList.shape, resultList.shape)

model = Sequential()
# input_dim取決於有多少features, 第一層須給定input_dim
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# loss:評分標準 / optimizer:優化器
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(inputList, resultList, epochs=100, batch_size=20)

scores = model.evaluate(inputList, resultList)

metrics = model.metrics_names
for s, m in zip(scores, metrics):
    print(f"{m} score={s}")
# print("scores=", scores)