# From the course: Modern Deep Learning in Python
# Link: https://www.udemy.com/data-science-deep-learning-in-theano-tensorflow/
# Dataset: https://www.kaggle.com/c/digit-recognizer/

# Introduces basic variables and functions in Tensorflow.


import numpy as np
import tensorflow as tf

# you have to specify the type
A = tf.placeholder(tf.float32, shape=(5, 5), name='A')

# but shape and name are optional
v = tf.placeholder(tf.float32)

# similar to dot in theano
w = tf.matmul(A, v)

with tf.Session() as session:
    # the values are fed in via the appropriately named argument "feed_dict"
    # v needs to be of shape=(5, 1) not just shape=(5, )
    # it's more like "real" matrix multiplication
    output = session.run(w, feed_dict={A: np.random.randn(5, 5), v: np.random.randn(5, 1)})
    print(output, type(output))

# A tf variable can be initialized with a numpy array or a tf array
# or more correctly, anything that can be turned into a tf tensor
shape = (2, 2)
x = tf.Variable(tf.random_normal(shape))
# x = tf.Variable(np.random.randn(2, 2))
t = tf.Variable(0) # a scalar

# need to "initialize" the variables first
init = tf.global_variables_initializer()

with tf.Session() as session:
    out = session.run(init)
    print(out)

    print(x.eval()) # similar to get_value() in Theano
    print(t.eval())

u = tf.Variable(20.0)
cost = u*u + u + 1.0

# One difference between Theano and TensorFlow is that one don't write the updates themselves in TensorFlow.
# Instead choose an optimizer that implements the algorithm
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

# let's run a session again
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    # weight update is automated but not the loop itself
    for i in range(12):
        session.run(train_op)
        print("i = %d, cost = %.3f, u = %.3f" % (i, cost.eval(), u.eval()))
