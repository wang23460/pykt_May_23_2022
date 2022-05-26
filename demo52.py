# use tensorflow
import tensorflow as tf

x = tf.Variable(1.)
with tf.GradientTape() as tape:
    y = 4 * x ** 2 + 2 * x + 3
    diff_x_1 = tape.gradient(y, x)
    print(f"dy/dx={diff_x_1}")

x2 = tf.Variable(tf.random.uniform((2, 2)))

with tf.GradientTape() as tape:
    y = 5 * x2 ** 2 + 4
    diff_x_2 = tape.gradient(y, x2)
    print("x2=\n", x2)
    print(f"dy/dx={diff_x_2}")

# 微積分
W = tf.Variable(tf.random.uniform((1, 1)))
b = tf.Variable(tf.zeros((1,)))
x3 = tf.random.uniform((1, 1))
with tf.GradientTape() as tape:
    y = tf.matmul(x3, W) + 2 * b
    grad_y_with_W_b = tape.gradient(y, [W, b])
    print(f"x3={x3[0][0]}")
    print(f"W={W.numpy()}")
    print(f"b={b.numpy()}")
    print(f"y={y[0][0]}")
    print(f"dy/dW={grad_y_with_W_b[0].numpy()[0][0]}")
    print(f"dy/db={grad_y_with_W_b[1].numpy()[0]}")