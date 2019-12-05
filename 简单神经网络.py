import tensorflow as tf
from numpy.random import RandomState

# 用numpy来生成一个模拟的数据集
# 定义训练数据batch的大小，在这里设置了一批批的
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))  # normol为正态分布，stddev是均值，seed是标准差
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的一个维度上使用None可以方便使用不同的batch大小。在训练时需要把数据分成比较小的batch，
# 但是在测试时，可以一次性使用全部数据。数据集比较小时可以这样做，大了可能会导致内存溢出。
# x为输入，y_为预测输出，还没有数据，先放在placeholder里，即用占位符表示
x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

# 定义神经网络前向传播过程，matmul为矩阵乘法
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 得到的y是一个数，通过sigmoid转化成一个0-1的数
y = tf.sigmoid(y)
# 定义损失函数为交叉熵
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
# 定义反向传播的算法，使得在当前batch下损失函数最小
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)  # 128组数据，每组两个x1和x2

# 定义一个Y规则，在这里x1+x2<1的样例都被认为是正样本，其他为负。
# 在这里使用0来表示负样本，1表示正样本：int里面为真就为1，假则为0
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# 准备工作都做好了，开始运行，创建一个会话Session来运行tf程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 初始化变量，global初始化所有变量

    sess.run(init_op)
    # 训练前先输出看下参数：w1是2*3矩阵，w2是3*1矩阵
    print(sess.run(w1))
    print(sess.run(w2))


    # 定义训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size（前面定义了为8）个样本进行训练，开始就是0-8，...一直到120-128
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 使用选取的这么多个样本来进行训练并更新参数，y是训练出来的预测值（有x给出就能计算y，所有字典里只用给出x和y_的值）
        # y_是真实值,feed_dict是一个字典，需要传值给占位符，如下的cross_entropy需要x和y_的值
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔1000轮计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training steps,cross entropy on all data is %s" % (i, total_cross_entropy))


    # 训练之后再次输出神经网络的参数值：
    print(sess.run(w1))
    print(sess.run(w2))

