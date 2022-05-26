import tensorflow as tf
import keras

print(tf.__version__)
print(keras.__version__)
t1 = tf.constant("hello tensorflow!")
print(t1)
print(t1.numpy())