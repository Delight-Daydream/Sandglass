import tensorflow as tf
import numpy as np

class image_learn:
    def __init__(self, ipath="./lab/t10k-images-idx3-ubyte", lpath="./lab/t10k-labels-idx1-ubyte", logpath="./log"):
        self.ipath = ipath
        self.lpath = lpath
        self.logpath = logpath

        self.ifile = open(self.ipath, "rb")
        self.imagic = int.from_bytes(self.ifile.read(4), 'big')
        self.icount = int.from_bytes(self.ifile.read(4), 'big')
        self.irows = int.from_bytes(self.ifile.read(4), 'big')
        self.icols = int.from_bytes(self.ifile.read(4), 'big')
        self.unitsize = self.irows * self.icols

        self.lfile = open(self.lpath, "rb")
        self.lmagic = int.from_bytes(self.lfile.read(4), 'big')
        self.lcount = int.from_bytes(self.lfile.read(4), 'big')

    def tfinit(self):
#        self.saver = tf.train.Saver(tf.global_variables())
        self.sess = tf.Session()

        self.x = tf.placeholder(tf.float32, [None, self.unitsize])

        self.W = tf.Variable(tf.zeros([self.unitsize, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.crossentropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices = [1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.crossentropy)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.saver = tf.train.Saver(tf.global_variables())

        self.ckpt = tf.train.get_checkpoint_state('./model')

        if self.ckpt and tf.train.checkpoint_exists(self.ckpt.model_checkpoint_path):
            print("Load model!")
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        else:
            print("Self init!")
            self.sess.run(tf.global_variables_initializer())

    def learn(self):
        idata = self.ifile.read(self.unitsize)
        ldata = self.lfile.read(1)

        _label_data = list()
        for i in range(10):
            _label_data.append(0)
        _label_data[int.from_bytes(ldata, 'big')] = 1
        label_data = list()
        label_data.append(_label_data)

        _image_data = list(idata)
        image_data = list()
        image_data.append(_image_data)

#        print("list size : ", len(data))
#        print("image_data : ", image_data)
#        print(self.sess.run(self.train_step, feed_dict={self.x: image_data, self.y_:[[0,1,2,3,4,5,6,7,8,9]]}))
        print(self.sess.run(self.train_step, feed_dict={self.x: image_data, self.y_: label_data}))
        print(self.sess.run(self.accuracy, feed_dict={self.x: image_data, self.y_: label_data}))

    def end(self):
        self.saver.save(self.sess, './model/network.ckpt')

#ifile = open("./lab/t10k-images-idx3-ubyte", "rb")
#lfile = open("./lab/t10k-labels-idx1-ubyte", "rb")

a = image_learn()

a.tfinit()
a.learn()
a.learn()
a.learn()
a.learn()
a.learn()
a.learn()
a.end()
