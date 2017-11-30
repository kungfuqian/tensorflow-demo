import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases    # 注意 inputs在前，Weights在后，有讲究，tf.matmul(inputs,Weights)
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return [outputs, Weights, biases]

# prepare data
x_data = np.linspace(-1,1,300)[:, np.newaxis]

#在指定的间隔内返回均匀间隔的数字。numpy.linspace(start, stop, num=300)
#  np.newaxis的功能是插入新维度

noise = np.random.normal(0, 0.05, x_data.shape)  # add noise
y_data = np.square(x_data) - 0.5 + noise


xs = tf.placeholder(tf.float32,[None,1]) # 随便输入多少个样本，属性为1
ys = tf.placeholder(tf.float32,[None,1])  #不要忘记加  tf.float32
# build first layer
# 定义隐藏层
[l1, _, _] = add_layer(xs,1,10,activation_function=tf.nn.relu) # 输入1个参数，隐藏层有10个神经元，激励函数用 relu
# 输出层
[prediction, Weights, biases] = add_layer(l1, 10, 1,activation_function=None)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init= tf.global_variables_initializer()         # 记住 global_variables_initializer

sess= tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(train_step, feed_dict={xs:x_data,ys:y_data})      #引用sess.run()时，若采用 placeholder,则需要feed_dict=...
    if step % 50 == 0:
        print(sess.run(loss, feed_dict={xs:x_data,ys:y_data}))       #引用sess.run()时，若采用 placeholder,则需要feed_dict=...
        print(sess.run([Weights, biases]))