
#################################################################################
#     sklearn.datasets 包提供了一些小的toy数据集。
# 为了评估数据特征(n_samples,n_features)的影响,
# 可以控制数据的一些统计学特性，产生人工数据。
#     sklearn对于不同的数据类型提供三种数据接口。
# dataset 产生函数和svmlight 加载器共享一个接口
# toy dataset和来自mldata.org上的数据都有比较复杂的结构
# sklearn包含一些不许要下载的toy数据集，见下表：
# 导入toy数据的方法 	                    介绍 	                任务 	数据规模
# load_boston() 	          加载和返回一个boston房屋价格的数据集 	回归 	506*13
# load_iris([return_X_y])   加载和返回一个鸢尾花数据集 	        分类 	150*4
# load_diabetes() 	      加载和返回一个糖尿病数据集 	        回归 	442*10
# load_digits([n_class])    加载和返回一个手写字数据集 	        分类 	1797*64
# load_linnerud() 	      加载和返回健身数据集                	多分类 	20
# 使用方法：
# from sklearn.datasets import load_linnerud
# linnerud = load_linnerud()
# linnerud.data
# linnerud.target
# linnerud.feature_names
# linnerud.target_names

# 数据集划分：sklearn.model_selection.train_test_split(*arrays, **options)
# *arrays：可以是列表、numpy数组、scipy稀疏矩阵或pandas的数据框
# test_size：可以为浮点、整数或None，默认为None
# ①若为浮点时，表示测试集占总样本的百分比
# ②若为整数时，表示测试样本样本数
# ③若为None时，test size自动设置成0.25
# train_size：可以为浮点、整数或None，默认为None
# ①若为浮点时，表示训练集占总样本的百分比
# ②若为整数时，表示训练样本的样本数
# ③若为None时，train_size自动被设置成0.75

# 数据预处理主要在sklearn.preprcessing包下
# 标签二值化LabelBinarizer
# 对于标称型数据来说，preprocessing.LabelBinarizer是一个很好用的工具。
# 比如可以把yes和no转化为0和1，或是把incident和normal转化为0和1。
#Example:
# Binary targets transform to a column vector
# >>> lb = preprocessing.LabelBinarizer()
# >>> lb.fit_transform(['yes', 'no', 'no', 'yes'])
# array([[1],
#        [0],
#        [0],
#        [1]])
#################################################################################
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)

def add_layer(inputs, in_size, out_size, layer_name,activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size,out_size],dtype=tf.float32))
    biases = tf.Variable(tf.zeros([1,out_size],dtype=tf.float32)+0.1)
    y = tf.matmul(inputs,Weight)+biases
    #dropout
    y = tf.nn.dropout(y,keep_prob)
    if activation_function == None:
        outputs = y
    else:
        outputs = activation_function(y)

    tf.summary.histogram( layer_name+'/outputs',outputs)
    return outputs

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,64])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(dtype=tf.float32)


# add output layer
l1 = add_layer(xs, 64, 100, 'l1', tf.nn.tanh)
prediction = add_layer(l1, 100, 10, 'l2', tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# summary writer goes in here
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('logs/train',sess.graph)
test_writer = tf.summary.FileWriter('logs/test',sess.graph)

sess.run(tf.global_variables_initializer())

for step in range(1001):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.6})
    if step % 50 == 0:
        # record loss
        train_result = sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        test_result = sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})

        train_writer.add_summary(train_result,step)
        test_writer.add_summary(test_result,step)
