# usage: tensorboard --logdir='logs/' to view the result with picture
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input, in_size, out_size, n_layer, activation_function = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram(layer_name+'/weights', W)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1,out_size]))
            tf.summary.histogram(layer_name + '/bias', bias)
        with tf.name_scope('y'):
            y = tf.matmul(input,W) + bias

        if activation_function == None:
            output = y
        else:
            output = activation_function(y)
        tf.summary.histogram(layer_name + '/output', output)
    return output

# prepare data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) + noise

#define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1], name='x_input')
    ys = tf.placeholder(tf.float32,[None,1], name='y_input')

l1 = add_layer(xs, 1, 10, n_layer= 1, activation_function=tf.nn.relu)
predict = add_layer(l1, 10, 1, n_layer= 2)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(predict - ys))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    optimize = tf.train.GradientDescentOptimizer(0.1)
    train = optimize.minimize(loss)


init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
fig.show()


sess = tf.Session()
sess.run(init)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
for step in range(2000):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})

    if step % 50 == 0:
        print(step, sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        predict_value = sess.run(predict, feed_dict={xs: x_data})
        lines = ax.plot(x_data, predict_value, 'r-', lw=5)

        plt.pause(0.1)
        ax.lines.remove(lines[0])
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,step)
