import tensorflow as tf
import numpy as np

class image_learn:
    def __init__(self, ipath="./lab/t10k-images-idx3-ubyte", lpath="./lab/t10k-labels-idx1-ubyte", logpath="./log"):
        self.ipath = ipath
        self.lpath = lpath
        self.logpath = logpath

        self.ifile = open(self.ipath, "rb")
        self.imagic = self.ifile.read(4)
        self.icount = int.from_bytes(self.ifile.read(4), 'big')
        self.irows = int.from_bytes(self.ifile.read(4), 'big')
        self.icols = int.from_bytes(self.ifile.read(4), 'big')
        self.unitsize = self.irows * self.icols

        self.lfile = open(self.lpath, "rb")
        self.lmagic = self.lfile.read(4)
        self.lcount = self.lfile.read(4)

    def tfinit(self):
#        self.saver = tf.train.Saver(tf.global_variables())
        self.sess = tf.Session()
        self.ckpt = tf.train.get_checkpoint_state('./model')
        if self.ckpt and tf.train.checkpoint_exists(self.ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

#        self.input = tf.placeholder(tf.uint8, [None, self.unitsize])
        self.x = tf.placeholder(tf.float32, [None, self.unitsize])
        print("info of self.x ", self.x)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.W = tf.Variable(tf.zeros([self.unitsize, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.res = tf.matmul(self.x, self.W)

        self.saver = tf.train.Saver(tf.global_variables())

    def learn(self):
        data = self.ifile.read(self.unitsize)
        
        list_data = list(data)
        """
        list_data = list()
        for i in list(range(self.unitsize)):
            tmp = list(self.ifile.read(1))
            list_data.append(tmp)
        """

        nparr = np.array(list_data)
        nparr.reshape(1, 784)

        test_data = list()
        test_data.append(list_data)
        test_data.append(list_data)
        print("test_data : ", test_data)
#        test_data.append(list_data)
#        test_data.append(list_data)
#        test_data.append(list_data)

#        print("list size : ", len(data))
#        print("list_data : ", list_data)
#        self.sess.run(self.res, feed_dict={self.x: list_data})
        self.sess.run(self.res, feed_dict={self.x: nparr})
        self.global_step = tf.add(self.global_step, tf.constant(1))

    def end(self):
        self.saver.save(self.sess, './model/network.ckpt', global_step=self.global_step)

    def dump(self):
        print(self.ipath)
        print(self.lpath)
        print(int.from_bytes(self.imagic, 'big'))
        print(self.icount)
        print(self.irows)
        print(self.icols)
        print(int.from_bytes(self.lmagic, 'big'))
        print(int.from_bytes(self.lcount, 'big'))

#ifile = open("./lab/t10k-images-idx3-ubyte", "rb")
#lfile = open("./lab/t10k-labels-idx1-ubyte", "rb")

a = image_learn()

a.dump()

a.tfinit()
#a.learn()
#a.learn()
#a.learn()
#a.learn()
#a.learn()
a.learn()
a.end()
