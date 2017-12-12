import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    y = tf.matmul(inputs, Weight) + biases

    if activation_function == None:
        outputs = y
    else:
        outputs = activation_function(y)
    return outputs


def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))    # tf.argmax(input, axis=None, name=None, dimension=None) axis：0表示按列，1表示按行 返回：Tensor  一般是行或列的最大值下标向量;
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))       # tf.cast(x, dtype, name=None)  此函数是类型转换函数  x：输入  dtype：转换目标类型  返回：Tensor
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# define placeholder for inputs to network

xs = tf.placeholder(tf.float32, [None,784])  #28x28 = 784
ys = tf.placeholder(tf.float32,[None,10])

# add output layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

#important step
sess.run(tf.global_variables_initializer())

for step in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train,feed_dict={xs:batch_xs,ys:batch_ys})
    if step % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
