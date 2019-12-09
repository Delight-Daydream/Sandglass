import tensorflow as tf
import struct

class image_learn:
    def __init__(self, ipath="./lab/t10k-images-idx3-ubyte", lpath="./lab/t10k-labels-idx1-ubyte", logpath="./log"):
        self.ipath = ipath
        self.lpath = lpath
        self.logpath = logpath

        self.ifile = open(self.ipath, "rb")
        self.imagic = self.ifile.read(4)
        self.icount = self.ifile.read(4)
        self.irows = self.ifile.read(4)
        self.icols = self.ifile.read(4)
        self.unitsize = self.irows * self.icols

        self.lfile = open(self.lpath, "rb")
        self.lmagic = self.lfile.read(4)
        self.lcount = self.lfile.read(4)

    def tfinit(self):
        self.sess = tf.Session()
        self.Input = tf.placeholder(tf.uint8, [None, self.unitsize])
        self.W = tf.Variable(tf.zeros[self.unitsize, 10])
        self.b = tf.Variable(tf.zeros[10])

    def learn(self):
        data = self.ifile.read(self.unitsize)

    def dump(self):
        print(self.ipath)
        print(self.lpath)
        print(int.from_bytes(self.imagic, 'big'))
        print(int.from_bytes(self.icount, 'big'))
        print(int.from_bytes(self.irows, 'big'))
        print(int.from_bytes(self.icols, 'big'))
        print(int.from_bytes(self.lmagic, 'big'))
        print(int.from_bytes(self.lcount, 'big'))

#ifile = open("./lab/t10k-images-idx3-ubyte", "rb")
#lfile = open("./lab/t10k-labels-idx1-ubyte", "rb")

a = image_learn()

a.dump()

a.tfinit()
