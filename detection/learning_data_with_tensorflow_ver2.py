import numpy as np
import tensorflow as tf
from mnist import load_mnist

# 1: データの取得
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

# 2: データの仕分け
xs = []
ts = []

for x, t in zip(x_train, t_train):
    if t == 1:
        xs.append(x)
        ts.append(np.array([1, 0]))

    elif t == 7:
        xs.append(x)
        ts.append(np.array([0, 1]))

xs = np.array(xs)
ts = np.array(ts)


# 3: 正解値の用意
t = tf.placeholder(tf.float32, [None, 2])

# 4: 入力層の値を用意
x1 = tf.placeholder(tf.float32, [None, 784])
w1 = tf.Variable(tf.random_normal([784,100], mean=0.0, stddev=0.05), dtype=tf.float32, name="w1")
b1 = tf.Variable(tf.zeros([100]), dtype=tf.float32, name="b1")

# 5: 入力層から中間層への順伝播
a1 = tf.matmul(x1, w1) + b1
x2 = tf.nn.relu(a1)

# 6: 中間層の値を用意
w2 = tf.Variable(tf.random_normal([100,2], mean=0.0, stddev=0.05), dtype=tf.float32, name="w2")
b2 = tf.Variable(tf.zeros([2]), dtype=tf.float32, name="b2")

# 7: 中間層から出力層への順伝播
a2 = tf.matmul(x2, w2) + b2
y = tf.nn.softmax(a2)

# 8: クロスエントロピーの導入
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y))

# 9: 誤差逆伝播を行うための準備
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 10: パラメータを保存するためのクラスの呼び出しとセッションの開始
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 11: 学習開始
for epoch in range(len(xs)):
    loss_ = sess.run(loss, feed_dict={x1: xs[epoch:epoch+1], t: ts[epoch:epoch+1]})
    print('epoch = ' + str(epoch) + ', loss = ' + str(loss_))
    sess.run(train_step, feed_dict={x1: xs[epoch:epoch+1], t: ts[epoch:epoch+1]})


# 12: パラメータの保存とセッションの切断
# saver.save(sess, "./model.ckpt")
""" メモ:
saver.save(sess, "./model.ckpt")を実行した時に作成されるファイル
・checkpoint
・model.ckpt.data-00000-of-00001
・model.ckpt.index
・model.ckpt.meta
"""
sess.close()