import numpy as np
from mnist import load_mnist

# 1: データの取得
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

# 2: データの仕分け
xs = []
ts = []

for x, t in zip(x_train, t_train):
    if t == 1:
        xs.append(x)
        ts.append(np.array([[1], [0]]))

    elif t == 7:
        xs.append(x)
        ts.append(np.array([[0], [1]]))


# 3: 必要な関数の定義
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_squared_error(y, t):
    return 0.5 * (np.sum((y - t)**2))


# 誤差関数をYで微分する処理を担当する関数
def delE_delY(y, t):
    return y - t


# 活性化関数YをA2で微分する処理を担当する関数
def delY_delA2(f, a2):
    return (1 - f(a2)) * f(a2)


# 活性化関数X2をA1で微分する処理を担当する関数
def delX2_delA1(f, a1):
    return (1 - f(a1)) * f(a1)


# 4: パラメーターWの準備
w1 = np.random.normal(0.0, 1, (10, 784))
w2 = np.random.normal(0.0, 1, (2, 10))


# 5: 学習開始
for x, t in zip(xs, ts):

    # 6: x1の形をw1.dot(x1)が計算できるように直してあげる
    x1 = x.reshape(784, 1)

    # 7:　入力層から中間層への順伝播
    a1 = w1.dot(x1)
    x2 = sigmoid(a1)

    # 8: 中間層から出力層への順伝播
    a2 = w2.dot(x2)
    y = sigmoid(a2)

    # 9: パラメーターの更新
    dEdA2 = delE_delY(y, t) * delY_delA2(sigmoid, a2)

    dEdA1 = (w2.T.dot(dEdA2) * delX2_delA1(sigmoid, a1)).dot(x1.T)
    w1 = w1 - dEdA1

    dEdW2 = dEdA2.dot(x2.T)
    w2 = w2 - dEdW2

print("学習終了")


# 10: パラメータwの保存(＊保存したくない場合はコメントアウトしておくと良い)
np.save("w1.npy", w1)
np.save("w2.npy", w2)


# 精度の確認
# 11: testデータから1と7のデータだけを取得
xs_test = []
ts_test = []

for x, t in zip(x_test, t_test):
    if t == 1:
        xs_test.append(x)
        ts_test.append(np.array([[1], [0]]))

    elif t == 7:
        xs_test.append(x)
        ts_test.append(np.array([[0], [1]]))


# 12: 実際のデータに対してどれくらいあったかを確認する
accuracy_cnt = 0
error_cnt = 0

for x, t in zip(xs_test, ts_test):
    x1 = x.reshape(784, 1)

    # 入力層 -> 中間層
    a1 = w1.dot(x1)
    x2 = sigmoid(a1)

    # 中間層 -> 出力層
    a2 = w2.dot(x2)
    y = sigmoid(a2)

    if np.argmax(y) == np.argmax(t):
        accuracy_cnt += 1

    else:
        error_cnt += 1

print("\n推論終了")
print("精度: ", accuracy_cnt / len(xs_test))
print("エラー: ", error_cnt, "/", len(xs_test))


