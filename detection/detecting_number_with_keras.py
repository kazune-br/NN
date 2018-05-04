import numpy as np
from keras.models import model_from_json
import os
from PIL import Image
from drawing_number import drawNumber

model = model_from_json(open('detection.json').read())
model.load_weights('detection.h5')

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

result = model.predict_classes(x)
answer = [1, 7]
print("answer: \n", answer[result[0]])
