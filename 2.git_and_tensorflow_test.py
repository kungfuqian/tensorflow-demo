###  the first time uses git and tensorflow 
# this is an example 

import tensorflow as tf
import numpy as np

# data prepare
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.3*x_data + 0.1

# tensorflow structure

Weights = tf.Variable(tf.random_uniform([1],0.29,0.31))
biases = tf.Variable(0.999*tf.ones([1]))

y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.71)
train= optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(Weights),sess.run(biases))

x = 100
y = sess.run(Weights)* x + sess.run(biases)
y2 = Weights*x + biases

print("y=",y,",y2=",y2)
