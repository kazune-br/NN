import os
import tensorflow as tf
import numpy as np
from PIL import Image
from drawing_number import drawNumber


"""Step1: 画像の生成から画像の取得を行う"""
# 画像ファイルの各ピクセルの値を正規化してあげる関数
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

# 好きな数字(1桁)を描いて保存 -> 保存ファイル名は'my_number.png'
drawNumber()

# 保存した数字の画像ファイルを呼び出す
n = len(os.listdir('./my_number'))
file = "./my_number/my_number{}.png".format(n)
img = np.array(Image.open(file).convert('L'))

# 画像ファイルをplaceholder"x1"の入力に渡してあげれるように変換する作業をしてあげる
x = min_max(img).reshape(1, 784)


"""Step2: 順伝播を行うために各変数の準備をする"""
# 入力層の値を用意
x1 = tf.placeholder(tf.float32, [None, 784])
w1 = tf.Variable(tf.random_normal([784,100], mean=0.0, stddev=0.05), dtype=tf.float32, name="w1")
b1 = tf.Variable(tf.zeros([100]), dtype=tf.float32, name="b1")

# 入力層から中間層への順伝播
a1 = tf.matmul(x1, w1) + b1
x2 = tf.sigmoid(a1)

# 中間層の値を用意
w2 = tf.Variable(tf.random_normal([100,2], mean=0.0, stddev=0.05), dtype=tf.float32, name="w2")
b2 = tf.Variable(tf.zeros([2]), dtype=tf.float32, name="b2")

# 中間層から出力層への順伝播
a2 = tf.matmul(x2, w2) + b2
y = tf.nn.sigmoid(a2)


"""Step3: Step2で準備した変数とtensorflowを用いて、Step1で取得した画像データの予測を行う"""
# セッションの開始と保存したパラメータの取得
sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "./model.ckpt")

# 実際に順伝播を行う
output = sess.run(y, feed_dict={x1: x})


"""Step4: 最終出力を確認する"""
answer = [1, 7]
print("output: \n", output)
print("\n", "answer: \n", answer[int(np.argmax(output))])
