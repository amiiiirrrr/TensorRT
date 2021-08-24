from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from utils.misc import get_paths_in_dir, decompose_labels_batchs, decompose_labels
import glob
from pathlib import Path
import imageio
import gc
from typing import List, Tuple, Optional

class CroppedImageDataGenerator(ImageDataGenerator):
    def __init__(self, featurewise_center = False, samplewise_center = False, 
                 featurewise_std_normalization = False, samplewise_std_normalization = False, 
                 zca_whitening = False, zca_epsilon = 1e-06, rotation_range = 0.0, width_shift_range = 0.0, 
                 height_shift_range = 0.0, brightness_range = None, shear_range = 0.0, zoom_range = 0.0, 
                 channel_shift_range = 0.0, fill_mode = 'nearest', cval = 0.0, horizontal_flip = False, 
                 vertical_flip = False, rescale = None, preprocessing_function = None, data_format = None, validation_split = 0.0, 
                 random_crop = None):
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, zca_whitening, zca_epsilon, rotation_range, width_shift_range, height_shift_range, brightness_range, shear_range, zoom_range, channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale, preprocessing_function, data_format, validation_split)
        # Random Crop
        assert random_crop == None or len(random_crop) == 2
        self.random_crop_size = random_crop

    def random_crop(self, original_img, seed=None):
        # Note: image_data_format is 'channel_last'
        assert original_img.shape[2] == 3
        if original_img.shape[0] < self.random_crop_size[0] or original_img.shape[1] < self.random_crop_size[1]:
            raise ValueError(f"Invalid random_crop_size : original = {original_img.shape}, crop_size = {self.random_crop_size}")
        
        height, width = original_img.shape[0], original_img.shape[1]
        dy, dx = self.random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return original_img[y:(y+dy), x:(x+dx), :]
    
    def flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None, 
            seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
        batches = super().flow(x, y, batch_size, shuffle, sample_weight, seed, save_to_dir, save_prefix, save_format, subset)
        while True:
            batch_x = next(batches)
            # Random crop
            if self.random_crop_size != None:
                x = np.zeros((batch_x.shape[0], self.random_crop_size[0], self.random_crop_size[1], 3))
                for i in range(batch_x.shape[0]):
                    x[i] = self.random_crop(batch_x[i], seed)
                batch_x = x
                del x
            # 返り値
            yield batch_x

    def get_datagen(self, img_dir_path: Path, label_dir_paths: Path, label_info, batch_size=16):
        img_paths = get_paths_in_dir(img_dir_path)
        img_paths.sort(key=lambda p: str(p))
        label_paths = get_paths_in_dir(label_dir_paths)
        label_paths.sort(key=lambda p: str(p))

        assert len(img_paths) == len(label_paths)

        h, w, _ = imageio.imread(str(img_paths[0]), as_gray=False, pilmode="RGB").shape
        imgs = np.zeros((batch_size, h, w, 3))
        labels = np.zeros((batch_size, h, w, 3))
        while True:
            seed = np.random.randint(0,10000000)
            l = list(np.random.randint(0, len(img_paths), batch_size))
            for i, j in enumerate(l):
                imgs[i,:,:,:] = imageio.imread(str(img_paths[j]), as_gray=False, pilmode="RGB")
                labels[i,:,:,:] = imageio.imread(str(label_paths[j]), as_gray=False, pilmode="RGB")
            img_gen = self.flow(imgs, batch_size=batch_size, seed=seed) 
            label_gen = self.flow(labels, batch_size=batch_size, seed=seed) 
            imgs_batch = next(img_gen)
            labels_batch = next(label_gen)
            labels_batch = decompose_labels_batchs(labels_batch, label_info)
            imgs_batch = imgs_batch.astype(np.float) / 255.
            labels_batch = labels_batch.astype(np.float) / 255.
            del img_gen, label_gen
            gc.collect()
            yield imgs_batch, labels_batch


# 画像をクロップ
def crop(img: np.ndarray, img_size: Tuple, center: Optional[Tuple] = None, left_top: Optional[Tuple] = None) -> np.ndarray:
    """

    :param img:
    :param center: y, x
    :param left_top: y, x
    :param img_size: height, width
    :return:
    """
    assert center is None or left_top is None
    height, width = img_size
    if center is None and left_top is None:
        center = img.shape[0] / 2, img.shape[1] / 2
        return img[int(center[0] - height / 2):int(center[0] + height / 2),
               int(center[1] - width / 2):int(center[1] + width / 2), :]
    elif center is not None:
        return img[int(center[0] - height / 2):int(center[0] + height / 2),
               int(center[1] - width / 2):int(center[1] + width / 2), :]
    elif left_top is not None:
        top, left = left_top
        bottom = top + img_size[0]
        right = left + img_size[1]
        return img[top:bottom, left:right, :]

# 画像のジェネレータ
def get_datagen(img_dir_path: Path, label_dir_paths: Path, label_info, img_size=(256, 256), batch_size=16, augmentation_rotation: bool = False, augmentation_brightness: bool = False, augmentation_horizontal_flip: bool = False, augmentation_vertical_flip: bool = False, augmentation_noise: bool = False):
    img_paths = get_paths_in_dir(img_dir_path)
    img_paths.sort(key=lambda p: str(p))
    label_paths = get_paths_in_dir(label_dir_paths)
    label_paths.sort(key=lambda p: str(p))

    assert len(img_paths) == len(label_paths)

    imgs = []
    labels = []

    while True:
        i = np.random.randint(0, len(img_paths))
        img_path = img_paths[i]
        label_path = label_paths[i]
        img = imageio.imread(str(img_path), as_gray=False, pilmode="RGB")
        label = imageio.imread(str(label_path), as_gray=False, pilmode="RGB")

        # ラベルの分割
        label = decompose_labels(label, label_info)

        # 正規化
        img = img.astype(np.float)
        label = label.astype(np.float)
        img /= 255.
        label /= 255.

        if augmentation_brightness:
            # Brightness
            brightness = 2 ** np.random.uniform(-1, 1.5)
            img *= brightness
            img = np.where(img > 1.0, 1.0, img)

        if augmentation_horizontal_flip:
            # Horizontal flip
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :]
                label = label[:, ::-1, :]

        if augmentation_vertical_flip:
            # Vertical flip
            if np.random.rand() > 0.5:
                img = img[::-1, :, :]
                label = label[::-1, :, :]

        if augmentation_noise:
            # Noise
            amount = 1e-3
            noise = amount * np.random.randn(*img.shape)
            img += noise

        # RandomCrop
        h, w, _ = img.shape
        top = 0 if h - img_size[0] == 0 else np.random.randint(0, h - img_size[0])
        left = 0 if w - img_size[1] == 0 else np.random.randint(0, w - img_size[1])
        cropped_img = crop(img, img_size, left_top=(top, left))
        cropped_label = crop(label, img_size, left_top=(top, left))

        imgs.append(cropped_img)
        labels.append(cropped_label)

        if len(imgs) == batch_size:
            imgs_batch = np.array(imgs)
            labels_batch = np.array(labels)
            imgs = []
            labels = []
            yield imgs_batch, labels_batch