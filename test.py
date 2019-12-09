# import tensorflow as tf

import struct

class image_learn:
    def __init__(self, ipath="./lab/t10k-images-idx3-ubyte", lpath="./lab/t10k-labels-idx1-ubyte"):
        self.ipath = ipath
        self.lpath = lpath
        self.ifile = open(self.ipath, "rb")

        self.imagic = self.ifile.read(4)
        self.icount = self.ifile.read(4)
        self.irows = self.ifile.read(4)
        self.icols = self.ifile.read(4)

        self.lfile = open(self.lpath, "rb")
        self.lmagic = self.lfile.read(4)
        self.lcount = self.lfile.read(4)

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
