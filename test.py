import tensorflow as tf
import numpy as np


#define layer
def add_layer(inputs, in_size, out_size, activation_function= None):

    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    diases = tf.Variable(tf.zeros([in_size,1]))

    result =tf.matmul(Weights,inputs) + diases

    if activation_function == None:
        outputs = result
    else:
        outputs = activation_function(result)
    return [outputs, Weights, diases]


#prepare data
x_data = tf.Variable(tf.random_normal([2,1000]))
truth_w = np.float32([[1,2],[3,4],[5,6],[7,8]])
truth_d = np.float32([[4],[3],[2],[1]])
#noise = tf.Variable(tf.random_normal([4,1], -0.5, 0.5))
y_data = tf.matmul(truth_w,x_data) + truth_d #+ noise


# build the structure
[l1,Weight,dias ] = add_layer(x_data, 4, 2, activation_function=tf.nn.relu)

[prediction,_ ,_ ] = add_layer(l1, 4, 4)

loss = tf.reduce_sum(tf.square(prediction - y_data),1)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(1001):
    sess.run(train)
    if step % 20 ==0:
        print(step,sess.run(loss),sess.run([Weight,dias]) )