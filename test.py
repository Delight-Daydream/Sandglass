# import tensorflow as tf

class image_learn:
    def __init__(self, ipath="./lab/t10k-images-idx3-ubyte", lpath="./lab/t10k-labels-idx1-ubyte"):
        self.ipath = ipath
        self.lpath = lpath
        self.ifile = open(self.ipath, "rb")
        self.lfile = open(self.lpath, "rb")
        self.idata = self.ifile.read()
        self.ldata = self.lfile.read()
        self.ifile.close()
        self.lfile.close()

    def dump(self):
        print(self.ipath)
        print(self.lpath)
        print(self.idata.count)
        print(self.ldata.count)

#ifile = open("./lab/t10k-images-idx3-ubyte", "rb")
#lfile = open("./lab/t10k-labels-idx1-ubyte", "rb")

a = image_learn()

a.dump()
