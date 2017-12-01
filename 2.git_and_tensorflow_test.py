import tensorflow as tf
import numpy as np

# 1.准备数据
x_data = np.float32(np.random.rand(2,100))  # 生成 2行 100列的矩阵   2代表两个属性，100为样本个数
y_data = np.matmul([0.1,0.6],x_data)+ 0.5


#print(x_data,y_data)

# 2.构造线性模型
diases = tf.Variable(tf.zeros([1]))
Weights = tf.Variable(tf.random_uniform([1,2]))  # 生成1*2 的随机矩阵
#print(Weights,diases)
y = tf.matmul(Weights,x_data) + diases

# 3. 求解模型
# 设置损失函数
loss = tf.reduce_mean(tf.square(y-y_data))

# 构造目标函数 ，选择梯度下降法
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 迭代目标： 最小化损失函数
train = optimizer.minimize(loss)


#################################################################
# 以下是用 tf来解决上面的任务

# 初始化所有变量
init=tf.global_variables_initializer()


## method 1
# sess = tf.Session()
# sess.run(init)
#
## 开始循环执行
# for step in range(200):
# 	sess.run(train)
# 	if step % 20 == 0:
# 		print(step,sess.run(Weights),sess.run(diases))
# sess.close()

# method 2
with tf.Session() as sess:
	sess.run(init)
	for step in range(301):
		sess.run(train)
		if step % 100 == 0:
			print(step, sess.run(Weights),sess.run(diases))
