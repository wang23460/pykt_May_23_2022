import tensorflow as tf

# use tensorflow ver1
tf.compat.v1.disable_eager_execution()

a = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
b = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)

with tf.compat.v1.Session() as session1:
    result1 = session1.run(c, feed_dict={
        a: [1, 2, 3],
        b: [4, 5, 6]
    })
    print(result1)