from keras.models import Sequential
from keras.layers.core import Dense, Activation
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
        ts.append(np.array([1, 0]))

    elif t == 7:
        xs.append(x)
        ts.append(np.array([0, 1]))

X = np.array(xs)
T = np.array(ts)

model = Sequential()
model.add(Dense(100, input_shape=X[0].shape))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

history = model.fit(X, T, batch_size=100, epochs=1, validation_data=None, verbose=1)


# 11: testデータから1と7のデータだけを取得
xs_test = []
ts_test = []

for x, t in zip(x_test, t_test):
    if t == 1:
        xs_test.append(x)
        ts_test.append(np.array([1, 0]))
        # ts_test.append(np.array([[1], [0]]))

    elif t == 7:
        xs_test.append(x)
        ts_test.append(np.array([0, 1]))
        # ts_test.append(np.array([[0], [1]]))

X_TEST = np.array(xs_test)
T_TEST = np.array(ts_test)

# evaluate model
score = model.evaluate(X_TEST, T_TEST, verbose=1)
print('test loss:', score[0])
print('test acc:', score[1])

# 学習結果の保存(Keras)
json_string = model.to_json()
open('detection.json', 'w').write(json_string)
model.save_weights('detection.h5')