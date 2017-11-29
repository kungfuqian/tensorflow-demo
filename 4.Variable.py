import tensorflow as tf

state = tf.Variable(1, name='counter')   # 设置初始值0，给变量名字
print(state.name)  # 输出 counter:0 输出名字counter，因为是第一个变量，所以为0

one = tf.constant(1) # 常亮

new_value = tf.add(state , one) # 加法
update = tf.assign(state, new_value)   # 将new_value的值赋值给state

init = tf.global_variables_initializer()   # must have if define variable

with tf.Session() as sess:         # 打开session
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))