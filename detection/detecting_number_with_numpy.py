import numpy as np
import os
from PIL import Image
from drawing_number import drawNumber


"""Step1: 順伝播で使用する関数を準備する"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""Step2: 画像の生成から画像の取得を行う"""
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


"""Step3: 順伝播を行うために各変数の準備をする"""
# データの準備
x1 = min_max(img).reshape(784, 1)

# learning_dataで取得したパラメーターを呼び出す
w1 = np.load('w1.npy')
w2 = np.load('w2.npy')

# 入力層 -> 中間層
a1 = w1.dot(x1)
x2 = sigmoid(a1)

# 中間層 -> 出力層
a2 = w2.dot(x2)
y = sigmoid(a2)


"""Step4: 最終出力を確認する"""
answer = [1, 7]
print("y: \n", y)
print("answer: \n", answer[int(np.argmax(y))])

