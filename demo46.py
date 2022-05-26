import tensorflow as tf
import numpy as np

# array
l1 = [5, 3, 8]
l2 = [3, -1, 2]
a1 = np.array(l1)
a2 = np.array(l2)
a3 = np.add(a1, a2)
print(a3)

# tensor
t1 = tf.constant(l1)
t2 = tf.constant(l2)
t3 = tf.add(t1, t2)
print(t3)
print(t3.numpy()) # to numpy array