import tensorflow as tf

#x = tf.placeholder(tf.float32)
x = tf.Variable(tf.zeros([3, 10]))
W = tf.Variable(tf.zeros([10, 3]))

A = tf.matmul(x, W)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#print(sess.run(A, feed_dict={x:}))
print(sess.run(A))
