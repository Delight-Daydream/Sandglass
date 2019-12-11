import tensorflow as tf

x = tf.placeholder(tf.float32, [1, 10])
#x = tf.placeholder(tf.float32)
W = tf.Variable(tf.ones([10, 3]))
b = tf.Variable(tf.ones([3]))

A = tf.matmul(x, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#testdata = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
testdata = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

print(sess.run(A, feed_dict={x: testdata}))

