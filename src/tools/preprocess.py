#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable-msg=logging-fstring-interpolation

"""
log_helper.py is a file to write methods which use for better log
"""

import logging
import imageio
import numpy as np
import tensorflow as tf
from utils import divide_image

__author__ = "Amir Mousavi"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"


def preprocess_image(target_img_path, train_img_size):
    """
    preprocess_image method is written for process image
    :param target_img_path: str
    :return:
        divided_image: Minutes of process
    """
    target_img = imageio.imread(str(target_img_path), as_gray=False, pilmode="RGB")
    target_float_img = target_img.astype(np.float32) / 255.
    divided_image =divide_image(target_float_img, train_img_size)
    # x = tf.constant(divided_image)
    divided_image = tf.convert_to_tensor(divided_image, dtype=tf.float32)
    return divided_image

def postprocess_image(target_img_path):
    """
    preprocess_image method is written for process image
    :param target_img_path: str
    :return:
        divided_image: Minutes of process
    """
    target_img = imageio.imread(str(target_img_path), as_gray=False, pilmode="RGB")
    target_float_img = target_img.astype(np.float32) / 255.
    divided_image =divide_image(target_float_img, train_img_size)
    # x = tf.constant(divided_image)
    divided_image = tf.convert_to_tensor(divided_image, dtype=tf.float32)
    return divided_image