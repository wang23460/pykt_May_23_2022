import tensorflow as tf


@tf.function
def add(p, q):
    return tf.math.add(p, q)


t1 = add([1, 2, 3], [4, 5, 6])
print(type(t1))
print(t1)