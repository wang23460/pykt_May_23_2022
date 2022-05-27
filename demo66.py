from keras.utils import to_categorical

origs = [4, 7, 16]
num_digits = 20
print(origs)
converted = to_categorical(origs, num_digits)
print(converted)