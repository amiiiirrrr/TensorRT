from tensorflow.keras.callbacks import Callback
from pathlib import Path
import numpy as np
import imageio
from typing import Tuple
from utils.misc import decompose_labels, divide_image_borderless, put_image_together_borderless, calc_IoU, binarize_label, decode_prob, make_overlay

# Segmentation用コールバック（エポック毎の推論とmIoUの算出）
class CallbackForSegmentation(Callback):
    def __init__(self, validation_img_path: Path, ground_truth_path: Path, label_info, model_path: Path, train_img_size: Tuple[int, int], period: int = 1):
        super().__init__()

        assert validation_img_path.is_file() and ground_truth_path.is_file()
        self.ground_truth = imageio.imread(str(ground_truth_path), as_gray=False, pilmode="RGB")
        self.ground_truth = self.ground_truth.astype(np.float32) / 255.
        self.ground_truth = decompose_labels(self.ground_truth, label_info=label_info)
        self.decoded_x = imageio.imread(str(validation_img_path), as_gray=False, pilmode="RGB")
        self.x = self.decoded_x.astype(np.float32) / 255.
        self.label_names = label_info.keys()
        self.label_info = label_info
        self.model_path = model_path
        self.image_path = model_path.joinpath("image")
        self.image_path.mkdir(exist_ok=True)
        self.period = period
        self.train_img_size = train_img_size
        self.iou = [f"epoch, {', '.join([label_name for label_name in label_info.keys()])}"]

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period != 0:
            return

        logs = logs or {}

        results = []
        divied_image = divide_image_borderless(self.x, self.train_img_size)
        for i in range(divied_image.shape[0]):
            tmp_y = self.model.predict(np.array([divied_image[i, :, :, :]]))[0]
            results.append(tmp_y)
        y = put_image_together_borderless(np.array(results), (self.x.shape[0], self.x.shape[1]))

        IoU = calc_IoU(ground_truth=binarize_label(self.ground_truth), inferenced=binarize_label(y))
        self.iou.append(f"{epoch}, {', '.join([str(iou_of_the_label) for iou_of_the_label in IoU])}")
        logs["val_mIoU"] = IoU.mean()

        for label_name, decoded_label, label_iou in zip(self.label_info.keys(), decode_prob(binarize_label(y)), IoU):
            overlay_img = make_overlay(self.decoded_x, decoded_label, label_color='r')
            imageio.imwrite(str(self.image_path / f"{str(epoch).zfill(3)}_{label_name}_{round(label_iou, 2)}.jpg"), overlay_img)
