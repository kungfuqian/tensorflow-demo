### first test for nural network and git 

import tensorflow as tf
import numpy as np

#create data

x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1*x_data + 0.3

### create tensorflow structure start  

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
diases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + diases
loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

### create tensorflow structure end ###


sess = tf.Session()
sess.run(init)


for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(Weights),sess.run(diases))
