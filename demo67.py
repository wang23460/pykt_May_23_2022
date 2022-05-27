#  flatten
import tensorflow as tf
import keras
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
FLATTEN_DIM = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
print(type(trainImages[0]))
print(np.unique(trainImages[0]))
trainImages = trainImages.astype(np.float32)
trainImages /= 255
testImages = testImages.astype(np.float32)
testImages /= 255
print(np.unique(trainImages[0]))
NUM_DIGITS = 10
trainLabels = keras.utils.to_categorical(train_labels, NUM_DIGITS)
testLabels = keras.utils.to_categorical(test_labels, NUM_DIGITS)

model = Sequential()
model.add(Dense(128, input_shape=(FLATTEN_DIM,), activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))
tb = TensorBoard(log_dir="logs/demo67", write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(trainImages, trainLabels, epochs=10, callbacks=[tb])

predictLabels = np.argmax(model.predict(testImages), axis=-1)
print(f"predict result:{predictLabels[:10]}")

loss, accuracy = model.evaluate(testImages, testLabels)
print(f"test images accuracy:{accuracy:.4f}")