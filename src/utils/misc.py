import numpy as np
import glob
from pathlib import Path
from typing import List, Tuple, Optional  # Ordered DictはPython3.7.2で追加
import cv2
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.keras import backend as K
import math
import imageio


# ディレクトリ内の画像パスのリストを生成
def get_paths_in_dir(dir_path: Path) -> List[Path]:
    paths = glob.glob(str(dir_path / '**' / '*.jpg'), recursive=True)
    paths.extend(glob.glob(str(dir_path / '**' / '*.JPG'), recursive=True))
    paths.extend(glob.glob(str(dir_path / '**' / '*.jpeg'), recursive=True))
    paths.extend(glob.glob(str(dir_path / '**' / '*.JPEG'), recursive=True))
    paths.extend(glob.glob(str(dir_path / '**' / '*.png'), recursive=True))
    paths.extend(glob.glob(str(dir_path / '**' / '*.PNG'), recursive=True))
    paths.extend(glob.glob(str(dir_path / '**' / '*.gif'), recursive=True))
    paths.extend(glob.glob(str(dir_path / '**' / '*.GIF'), recursive=True))
    paths.extend(glob.glob(str(dir_path / '**' / '*.bmp'), recursive=True))
    paths.extend(glob.glob(str(dir_path / '**' / '*.BMP'), recursive=True))
    return [Path(path) for path in paths]


# np.float32 -> np.uint8へ変換
def decode_img(x: np.ndarray) -> np.ndarray:
    x *= 255
    x = x.astype(np.uint8)
    return x


def decode_prob(y: np.ndarray) -> List[np.ndarray]:
    y = decode_img(y)
    result = []
    for i in range(y.shape[2]):
        y_single = y[:, :, i]
        y_single = np.expand_dims(y_single, axis=-1)
        result.append(y_single)
    return result


# ラベルをオーバーレイ合成
def make_overlay(img: np.ndarray, label: np.ndarray, label_color='b'):
    if label_color == "b":
        label = np.concatenate([np.zeros((label.shape[0], label.shape[1], 2)).astype(np.uint8), label], axis=2)
    elif label_color == "g":
        label = np.concatenate([np.zeros((label.shape[0], label.shape[1], 1)).astype(np.uint8), label,
                                np.zeros((label.shape[0], label.shape[1], 1)).astype(np.uint8)], axis=2)
    elif label_color == 'r':
        label = np.concatenate([label, np.zeros((label.shape[0], label.shape[1], 2)).astype(np.uint8)], axis=2)

    img = img.astype(np.float32) / 255.
    label = label.astype(np.float32) / 255.
    overlay = cv2.addWeighted(img, 0.7, label, 0.3, 0)
    overlay *= 255.
    return overlay.astype(np.uint8)


def decompose_labels_batchs(img: np.ndarray, label_info) -> np.ndarray:
    result_batchs = np.zeros((img.shape[0], img.shape[1], img.shape[2], len(label_info)))
    for i in range(img.shape[0]):
        result_batchs[i,:,:,:] = decompose_labels(img[i,:,:,:], label_info)
    return result_batchs


# RGB形式で出力されたラベル画像を1チャンネルの複数画像に分解
def decompose_labels(img: np.ndarray, label_info) -> np.ndarray:
    result = np.zeros((img.shape[0], img.shape[1], len(label_info)))
    for i, (_, color) in enumerate(label_info.items()):
        result[:, :, i] = decompose_label(img, color)
    return result


def decompose_label(img: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    if color == (0, 0, 0):
        img = np.where(img == 0, 1, img)
        color = (1, 1, 1)

    bgrLower = np.array([color[0], color[1], color[2]])
    bgrUpper = np.array([color[0], color[1], color[2]])  # 抽出する色の上限(BGR)
    img_mask = cv2.inRange(img, bgrLower, bgrUpper)  # BGRからマスクを作成
    result = cv2.bitwise_and(img, img, mask=img_mask)  # 元画像とマスクを合成
    result = result > 0
    new_result = np.zeros((result.shape[0], result.shape[1]))
    new_result[:, :] = result[:, :, 0] + result[:, :, 1] + result[:, :, 2]
    new_result = np.where(new_result > 0, 1., 0.)
    new_result *= 255
    new_result = new_result.astype(np.uint8)
    return new_result


# 重み付き交差エントロピーに使う値を計算
def calc_weights_of_crossentropy(dataset_path: Path, label_info) -> np.ndarray:
    train_label_paths = get_paths_in_dir(dataset_path.joinpath("train").joinpath("label"))
    pixcount = np.zeros((len(label_info),))
    for train_label_path in train_label_paths[:]:
        train_label = imageio.imread(str(train_label_path), as_gray=False, pilmode="RGB")
        train_label = decompose_labels(img=train_label, label_info=label_info)
        # 正規化
        train_label = train_label.astype(np.float)
        train_label /= 255.
        train_labels = []
        train_labels.append(train_label)
        train_labels = np.array(train_labels)
        # 各ラベルの出現ピクセル数
        pixcount += np.count_nonzero(train_labels, axis=(0, 1, 2))

    weights = 1. / (pixcount + 1e-4)
    weights /= weights.min()
    return weights


# 出力を2値化する
def binarize_label(img: np.ndarray) -> np.ndarray:
    result = np.zeros(img.shape, dtype=np.float32)
    max_val = np.max(img, axis=2)
    n_channel = img.shape[2]
    for i in range(n_channel):
        result[:, :, i] = max_val
    return (img == result).astype(np.float32)


# IOUを算出
def calc_IoU(ground_truth, inferenced) -> np.ndarray:
    # TODO: 0~1か0~255を判定
    # TODO: 2値化済みかを判定
    ground_truth = ground_truth.astype(np.uint8)
    inferenced = inferenced.astype(np.uint8)
    IoU = (ground_truth & inferenced).sum(axis=(0, 1)) / ((ground_truth | inferenced).sum(axis=(0, 1)) + 1e-4)
    return IoU


# 画像を学習時のサイズに分割
def divide_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """

    :param img:
    :param target_size: y ,x
    :return:
    """
    result = []
    img_y, img_x, _ = img.shape
    target_y, target_x = target_size
    n_x = math.ceil(img_x / target_x)
    n_y = math.ceil(img_y / target_y)
    for i in range(n_x):
        for j in range(n_y):
            # 余白埋め
            margin_y = max(0, (j+1)*target_y - img_y)
            margin_x = max(0, (i+1)*target_x - img_x)
            tmp_img = img[j*target_y: (j+1)*target_y, i*target_x: (i+1)*target_x, :]
            tmp_img = np.pad(tmp_img, [(0, margin_y), (0, margin_x), (0, 0)], 'constant')
            result.append(tmp_img.astype(np.float32))
    return np.array(result)


# 元に戻す
def put_image_together(imgs: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    img_y, img_x, img_c = imgs[0, :, :, :].shape
    target_y, target_x = target_size
    n_x = math.ceil(target_x / img_x)
    n_y = math.ceil(target_y / img_y)

    assert len(imgs) == n_x * n_y

    result = np.zeros((target_y, target_x, img_c))
    for i in range(n_x):
        for j in range(n_y):
            x = i * img_x
            y = j * img_y
            margin_x = max(0, x + img_x - target_x)
            margin_y = max(0, y + img_y - target_y)
            tmp_img = imgs[n_y * i + j, :, :, :]
            tmp_img = tmp_img[0: img_y - margin_y, 0: img_x - margin_x, :]  # 余白をトリミング
            result[y: y + img_y - margin_y, x: x + img_x - margin_x, :] = tmp_img

    return result.astype(np.float32)

# 画像を学習時のサイズに分割(サイズのうち，中心の部分を用いる．)
def divide_image_borderless(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    img_y, img_x, _ = img.shape
    target_y, target_x = target_size
    img_flip = np.flip(img)
    img_ud = np.flipud(img)
    img_lr = np.fliplr(img)
    upper_down_img = np.concatenate([img_flip, img_ud, img_flip], axis = 1)
    middle_img = np.concatenate([img_lr, img, img_lr], axis = 1)
    margined_image = np.concatenate([upper_down_img, middle_img, upper_down_img], axis = 0)
    y_loop = math.ceil(img_y / target_y * 2)
    x_loop = math.ceil(img_x / target_x * 2)
    results = [margined_image[int(img_y+(2*j-1)*target_y/4):int(img_y+(2*j+3)*target_y/4), int(img_x+(2*i-1)*target_x/4):int(img_x+(2*i+3)*target_x/4),].astype(np.float32) for i in range(x_loop) for j in range(y_loop)]
    return np.array(results)

# 元に戻す
def put_image_together_borderless(imgs: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    _, img_y, img_x, img_c = imgs.shape
    target_y, target_x = target_size

    img_y = int(img_y / 2)
    img_x = int(img_x / 2)

    n_x = math.ceil(target_x / img_x)
    n_y = math.ceil(target_y / img_y)

    assert len(imgs) == n_x * n_y

    result = np.zeros((target_y*2, target_x*2, img_c))
    for i in range(n_x):
        for j in range(n_y):
            x = i * img_x
            y = j * img_y
            tmp_img = imgs[n_y * i + j, :, :, :]
            #　中心部分のみを用いる
            tmp_img = tmp_img[int(img_y / 2): int(img_y * 3 / 2), int(img_x / 2): int(img_x * 3 / 2), :]
            result[y: y + img_y, x: x + img_x, :] = tmp_img
    # データを良い形に整形する．
    result = result[0:target_y,0:target_x,:]
    return result.astype(np.float32)


def weighted_crossentropy_wrapper(class_weights, eps=1e-7):
    def weighted_cross_entropy(onehot_labels, output):
        output = K.clip(output, eps, 1.)
        loss = - tf.reduce_mean(class_weights * onehot_labels * tf.math.log(output))
        return loss

    return weighted_cross_entropy


def focal_loss_wrapper(class_weights):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = K.mean(class_weights[1] * K.pow(1. - pt_1, class_weights[0]) * K.log(1. - pt_0))
        print(loss)
        return loss

    return focal_loss_fixed
