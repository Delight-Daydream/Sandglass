import tensorflow as tf

ifile = open("./lab/t10k-images-idx3-ubyte", "rb")
lfile = open("./lab/t10k-labels-idx1-ubyte", "rb")

imagic = int.from_bytes(ifile.read(4), 'big')
icount = int.from_bytes(ifile.read(4), 'big')
irows = int.from_bytes(ifile.read(4), 'big')
icols = int.from_bytes(ifile.read(4), 'big')
unitsize = irows * icols

lmagic = int.from_bytes(lfile.read(4), 'big')
lcount = int.from_bytes(lfile.read(4), 'big')

x = tf.placeholder(tf.float32, [None, unitsize])
#W = tf.Variable(tf.zeros([unitsize, 10]))
#b = tf.Variable(tf.zeros([10]))
W = tf.Variable(tf.random_normal([unitsize, 10]))
b = tf.Variable(tf.random_normal([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
opt = tf.train.GradientDescentOptimizer(0.5)
train_step = opt.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(W))

for i in range(1):
    image_list = list()
    label_list = list()

    for j in range(1):
        idata = ifile.read(unitsize)
        ldata = lfile.read(1)

        # one hot encoding
        label_vec = list()
        for k in range(10):
            label_vec.append(0)
        label_vec[int.from_bytes(ldata, 'big')] = 1
        label_list.append(label_vec)

        image_vec = list(idata)
        image_list.append(image_vec)

    sess.run(train_step, feed_dict={x: image_list, y_: label_list})
    print(sess.run(W))

ifile.close()
lfile.close()
