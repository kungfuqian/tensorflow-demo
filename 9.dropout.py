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
