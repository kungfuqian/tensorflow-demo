#tf.placeholder(dtype, shape=None, name=None)
#dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
#shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
#name：名称。
# 用于得到传递进来的真实的训练样本：不必指定初始值，可在运行时，通过 Session.run 的函数的 feed_dict 参数指定；这也是其命名的原因所在，仅仅作为一种占位符；


import tensorflow as tf

input1 = tf.placeholder(tf.float32)  #tf.placeholder(dtype, shape=None, name=None)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)  # 乘法运算

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))  #以字典形式