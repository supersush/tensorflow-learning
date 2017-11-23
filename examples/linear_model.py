import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

b = tf.Variable(tf.zeros([1]))
# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# W = tf.Variable(tf.glorot_uniform_initializer([1, 2]))
W=tf.get_variable("W",dtype=tf.float32,shape=[1,2])
y = tf.matmul(W, x_data) + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
print sess.run(init)
print x_data
print y_data

print sess.run(W)
print sess.run(b)
for step in xrange(0, 401):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b),sess.run(loss)
