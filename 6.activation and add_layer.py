import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
x_data = np.linspace(4, 400,300)[:, np.newaxis]
# data regulation
mean = np.mean(x_data)
max = np.max(x_data)
min = np.min(x_data)

x_data_new = (x_data - mean)/(max - min)

#在指定的间隔内返回均匀间隔的数字。numpy.linspace(start, stop, num=300)
#  np.newaxis的功能是插入新维度

noise = np.random.normal(0, 0.05, x_data.shape)  # add noise
y_data = np.square(x_data_new) - 0.5 + noise


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


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data_new,y_data)
plt.ion()
plt.show()


for step in range(1000):
    sess.run(train_step, feed_dict={xs:x_data_new,ys:y_data})      #引用sess.run()时，若采用 placeholder,则需要feed_dict=...
    if step % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        #print(sess.run(loss, feed_dict={xs:x_data_new,ys:y_data}))       #引用sess.run()时，若采用 placeholder,则需要feed_dict=...
        #print(sess.run([Weights, biases]))
        prediction_value = sess.run(prediction, feed_dict={xs: x_data_new})
        lines = ax.plot(x_data_new,prediction_value,'r-',lw=5)

        plt.pause(0.1)




# python中try/except/else/finally语句的完整格式如下所示：
# try:
#      Normal execution block
# except A:
#      Exception A handle
# except B:
#      Exception B handle
# except:
#      Other exception handle
# else:
#      if no exception,get here
# finally:
#      print("finally")
#
#      python中try /except / else / finally语句的完整格式如下所示：
#      try:
#          Normal
#          execution
#          block
#      except A:
#          Exception
#          A
#          handle
#      except B:
#          Exception
#          B
#          handle
#      except:
#          Other
#          exception
#          handle
#      else:
#          if no exception, get here
#      finally:
#          print("finally")

# 说明：
# 正常执行的程序在try下面的Normal execution block执行块中执行，在执行过程中如果发生了异常，则中断当前在Normal execution block中的执行跳转到对应的异常处理块中开始执行；
# python从第一个except X处开始查找，如果找到了对应的exception类型则进入其提供的exception handle中进行处理，如果没有找到则直接进入except块处进行处理。except块是可选项，如果没有提供，该exception将会被提交给python进行默认处理，处理方式则是终止应用程序并打印提示信息；
# 如果在Normal execution block执行块中执行过程中没有发生任何异常，则在执行完Normal execution block后会进入else执行块中（如果存在的话）执行。
# 无论是否发生了异常，只要提供了finally语句，以上try/except/else/finally代码块执行的最后一步总是执行finally所对应的代码块。
# 需要注意的是：
# 1.在上面所示的完整语句中try/except/else/finally所出现的顺序必须是try-->except X-->except-->else-->finally，即所有的except必须在else和finally之前，else（如果有的话）必须在finally之前，而except X必须在except之前。否则会出现语法错误。