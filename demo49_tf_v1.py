import tensorflow as tf

# use tensorflow ver1
tf.compat.v1.disable_eager_execution()

# 三角形公式 (海龍公式)
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]

    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


with tf.compat.v1.Session() as session1:
    area = computeArea(tf.compat.v1.constant([
        [3.0, 4.0, 5.0],
        [6.0, 6.0, 6.0, ],
        [2.5, 4.1, 4.0]
    ]))
    result1 = session1.run(area)
    print(result1)