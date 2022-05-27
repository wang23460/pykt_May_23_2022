from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense,Flatten

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000, 28, 28, 1)).astype('float32')
X_test = X_test.reshape((10000, 28, 28, 1)).astype('float32')
X_train /= 255
X_test /= 255
# print(X_train[0])
# print(X_test[0])
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)



def model():
    m = Sequential()
    m.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dense(10, activation='softmax'))
    m.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
    m.summary()
    return m


model = model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100)

scores = model.evaluate(X_test, y_test)
print(f"CNN Error:{100 - scores[1] * 100}%")