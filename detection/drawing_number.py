import cv2
import numpy as np
import os


# OpenCVのマウスイベントを扱うためのクラス
class CVMouseEvent:
    def __init__(self, press_func=None, drag_func=None, release_func=None):
        self._press_func = press_func
        self._drag_func = drag_func
        self._release_func = release_func

        self._is_drag = False

    # Callback登録関数
    def setCallBack(self, win_name):
        cv2.setMouseCallback(win_name, self._callBack)

    def _doEvent(self, event_func, x, y):
        if event_func is not None:
            event_func(x, y)

    def _callBack(self, event, x, y, flags, param):
        # マウス左ボタンが押された時の処理
        if event == cv2.EVENT_LBUTTONDOWN:
            self._doEvent(self._press_func, x, y)
            self._is_drag = True

        # マウス左ドラッグ時の処理
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._is_drag:
                self._doEvent(self._drag_func, x, y)

        # マウス左ボタンが離された時の処理
        elif event == cv2.EVENT_LBUTTONUP:
            self._doEvent(self._release_func, x, y)
            self._is_drag = False


# 描画用の空画像作成
def emptyImage():
    # return np.zeros((512, 512, 3), np.uint8)
    return np.zeros((512, 512, 1), np.uint8)


# シンプルなマウス描画のデモ
def drawNumber():
    img = emptyImage()
    color = (255, 255, 255)

    # ドラッグ時に描画する関数の定義
    def brushPaint(x, y):
        cv2.circle(img, (x, y), 10, color, -1)  # (画像, 中心座標, 半径, 色, 線の太さ)

    win_name = 'Drawing Number'
    cv2.namedWindow(win_name)

    # CVMouseEventクラスによるドラッグ描画関数の登録
    mouse_event = CVMouseEvent(drag_func=brushPaint)
    mouse_event.setCallBack(win_name)

    while(True):
        cv2.imshow(win_name, img)

        key = cv2.waitKey(30) & 0xFF

        # キーボード入力で"r"が入力されたらリセット
        if key == ord('r'):
            img = emptyImage()

        # キーボード入力で"q"が入力されたらそのまま終了
        elif key == ord('q'):
            break

        # キーボード入力で"s"が入力されたら画像ファイルを保存して終了
        elif key == ord('s'):
            img = cv2.resize(img, (28, 28))
            n = len(os.listdir('./my_number')) + 1
            cv2.imwrite('./my_number/my_number{}.png'.format(n), img)
            break

    cv2.destroyAllWindows()


# drawNumber()

