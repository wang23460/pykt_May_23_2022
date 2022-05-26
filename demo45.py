import tensorflow as tf

# 呼叫第一版tensorflow
tf.compat.v1.disable_eager_execution()
t1 = tf.constant("hello tensorflow!")
print(t1)
#tensorflow 第一版 需用session來啟動
session1 = tf.compat.v1.Session()
print(session1.run(t1))
session1.close()