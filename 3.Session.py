# session.run() 每run一下，tensorflow 执行一次tensorflow结构

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 忽略SSE4.1, SSE4.2, AVX, AVX2, FMA的警告

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
		       [2]])
 
product = tf.matmul(matrix1,matrix2)  # matrix multiply == np.dot(m1,m2)

# ctrl + / : batch annotation code

# # method 1
# sess = tf.Session()
# result = sess.run(product)
#
# print(result)
# sess.close()

#method 2

with tf.Session() as sess:
	result2 = sess.run(product)
	print(result2)
