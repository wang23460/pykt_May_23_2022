import numpy
from keras.datasets import imdb
from matplotlib import pyplot

(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
print(X.shape)
print(y.shape)

print(numpy.unique(y, return_counts=True))

print(X[0])
print(len(numpy.unique(numpy.hstack(X))))

result = [len(x) for x in X]
print(numpy.mean(result), numpy.std(result))

pyplot.subplot(121)
pyplot.boxplot(result)
pyplot.subplot(122)
pyplot.hist(result)
pyplot.show()