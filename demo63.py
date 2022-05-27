scores = [3.0, 1.0, 2.0]
import numpy as np
import tensorflow as tf


def manualSoftmax(x):
    ax = np.array(x)
    return np.exp(ax) / np.sum(np.exp(ax), axis=0)

# 透過softmax讓懸殊比例更懸殊
print(manualSoftmax(scores))
print(tf.nn.softmax(scores).numpy())