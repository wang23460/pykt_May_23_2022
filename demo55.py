# 手刻
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


# 用tf判別點是0/1
num_sample_per_class = 1000

negative_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_sample_per_class)
positive_samples = np.random.multivariate_normal(mean=[-3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_sample_per_class)
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
targets = np.vstack((np.zeros((num_sample_per_class, 1), dtype="float32"),
                     np.ones((num_sample_per_class, 1), dtype="float32")))

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.figure()

input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(output_dim, ))


def model(inputs):
    return tf.matmul(inputs, W) + b


def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


learning_rate = 0.1


def training_steps(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(predictions, targets)
    grad_loss_wrt_w, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_w * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    print(f"..W={W.numpy()[0], W.numpy()[1]},b={b.numpy()[0]}")
    return loss


for step in range(40):
    loss = training_steps(inputs, targets)
    print(f"loss step{step}, value={loss:.4f}")

predictions = model(inputs)
x = np.linspace(-4, 4, 100)
# W[0]x+W[1]y+b=0
# W[1]y = -W[0]x-b
y1 = -W[0] / W[1] * x + (0 - b) / W[1]
y2 = -W[0] / W[1] * x + (0.5 - b) / W[1]
y3 = -W[0] / W[1] * x + (1 - b) / W[1]
plt.plot(x, y1, "--b")
plt.plot(x, y2, "-r")
plt.plot(x, y3, "--b")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()