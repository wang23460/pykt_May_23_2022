import tensorflow as tf

time = tf.Variable(5.)
# 微分
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position, time)
    print("speed=", speed)
# 加速器
accelerator = outer_tape.gradient(speed, time)
print("accelerator=", accelerator)