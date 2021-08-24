import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # コメントアウトを外すとGPUを無効化
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0" #GPU

###GeForce GTX 1660 config###
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.models import load_model, Model
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from utils import decode_prob, weighted_crossentropy_wrapper, \
					make_overlay, calc_IoU, binarize_label, \
					get_paths_in_dir, divide_image, put_image_together,\
					decompose_labels, divide_image_borderless,\
					put_image_together_borderless
import numpy as np
import collections
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
import imageio
from threading import Thread
from tqdm import tqdm
import time
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.preprocessing import image
from configuration import BaseConfig


def main32():
	# load pre-trained model
	weight_path = Path("/home/ubuntu/Desktop/o-semantic-segmentation-feature-batch/weights/test2/final_model.h5")
	# weight_path = Path("/home/ubuntu/Desktop/o-semantic-segmentation-feature-batch/src/saved_model")
	# inference_targetの中のフォルダ名
	target_name = "0630"
	target_dir_path = Path(__file__).resolve().parent.parent.joinpath("inference_target", target_name)
	
	# assert weight_path.is_file() and target_dir_path.is_dir()
	# model: Model = load_model(str(weight_path),
	# custom_objects={"weighted_cross_entropy": weighted_crossentropy_wrapper([1., 60., 60.]), 'tf': tf})

	# model = load_model(weight_path)
	# model.save('saved_model/') 

	# model = tf.keras.models.load_model('saved_model/')
	result_name = "0630"
	result_path = Path(__file__).resolve().parent.parent.joinpath("inference_result", result_name)

	# model.summary()
	batch_size = 14
	img_shape=(512, 2048, 3)
	train_img_size=(512, 2048)
	label_info = collections.OrderedDict()
	label_info["background"] = (0, 0, 0)
	label_info["kizu"] = (128, 0, 0)

	

	

	result_path.mkdir(exist_ok=True)
	image_paths = get_paths_in_dir(target_dir_path)
	image_paths.sort()


	# print('Converting to TF-TRT FP32...')
	# conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
	# max_workspace_size_bytes=8000000000)

	# converter = trt.TrtGraphConverterV2(input_saved_model_dir=str(weight_path),
	# conversion_params=conversion_params)
	# converter.convert()
	# converter.save(output_saved_model_dir='saved_model_TFTRT_FP32')
	# print('Done Converting to TF-TRT FP32')

	saved_model_loaded = tf.saved_model.load('saved_model_TFTRT_FP32', tags=[tag_constants.SERVING])
	signature_keys = list(saved_model_loaded.signatures.keys())
	print(signature_keys)

	infer = saved_model_loaded.signatures['serving_default']
	print(infer.structured_outputs)
	print(infer)

	# labeling = infer(x)

	for target_img_path in tqdm(image_paths):
		target_img = imageio.imread(str(target_img_path), as_gray=False, pilmode="RGB")
		target_float_img = target_img.astype(np.float32) / 255.
		divided_image =divide_image(target_float_img, train_img_size)
		# x = tf.constant(divided_image)
		divided_image = tf.convert_to_tensor(divided_image, dtype=tf.float32)
		# print("##################################################### divided_image", x.shape)
		print("##################################################### divided_image", divided_image[1])
		t1 = time.time()
		# inputs = {}
		# inputs['input_1']=x
		# results = model.predict(np.array(divided_image),batch_size=batch_size)
		outputs = infer(input_1=divided_image[1:3])
		# outputs['activation_81']
		t2 = time.time()
		elapsed_time = t2-t1
		print(f"########## 経過時間_batch{batch_size}：{elapsed_time}")
		# y = put_image_together(outputs, (target_img.shape[0], target_img.shape[1]))

		# for label_name, decoded_label in zip(label_info.keys(), decode_prob(binarize_label(y))):
		# 	overlay_img = make_overlay(target_img, decoded_label, label_color='r')
		# 	Thread(target=imageio.imwrite, args=(str(result_path / f"label_{target_img_path.stem}_{label_name}.jpg"), overlay_img), daemon=False).start()
def main16():
	# load pre-trained model
	# weight_path = Path("/home/ubuntu/Desktop/o-semantic-segmentation-feature-batch/weights/test2/final_model.h5")
	weight_path = Path("/home/ubuntu/Desktop/o-semantic-segmentation-feature-batch/src/saved_model")
	# inference_targetの中のフォルダ名
	target_name = "0630"
	target_dir_path = Path(__file__).resolve().parent.parent.parent.joinpath("inference_target", target_name)
	
	# assert weight_path.is_file() and target_dir_path.is_dir()
	# model: Model = load_model(str(weight_path),
	# custom_objects={"weighted_cross_entropy": weighted_crossentropy_wrapper([1., 60., 60.]), 'tf': tf})

	# model = load_model(weight_path)
	# model.save('saved_model/') 

	# model = tf.keras.models.load_model('saved_model/')
	result_name = "0630"
	result_path = Path(__file__).resolve().parent.parent.parent.joinpath("inference_result", result_name)

	# model.summary()
	batch_size = 14
	img_shape=(512, 2048, 3)
	train_img_size=(512, 2048)
	label_info = collections.OrderedDict()
	label_info["background"] = (0, 0, 0)
	label_info["kizu"] = (128, 0, 0)

	
	result_path.mkdir(exist_ok=True)
	image_paths = get_paths_in_dir(target_dir_path)
	image_paths.sort()


	print('Converting to TF-TRT FP16...')
	conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
	precision_mode=trt.TrtPrecisionMode.FP16,
	max_workspace_size_bytes=8000000000)
	converter = trt.TrtGraphConverterV2(
	input_saved_model_dir=str(weight_path), conversion_params=conversion_params)
	converter.convert()
	converter.save(output_saved_model_dir='saved_model_TFTRT_FP16')
	print('Done Converting to TF-TRT FP16')

	saved_model_loaded = tf.saved_model.load('saved_model_TFTRT_FP16', tags=[tag_constants.SERVING])
	signature_keys = list(saved_model_loaded.signatures.keys())
	print(signature_keys)

	infer = saved_model_loaded.signatures['serving_default']
	print(infer.structured_outputs)
	print(infer)

	# labeling = infer(x)

	for target_img_path in tqdm(image_paths):
		target_img = imageio.imread(str(target_img_path), as_gray=False, pilmode="RGB")
		target_float_img = target_img.astype(np.float32) / 255.
		divided_image =divide_image(target_float_img, train_img_size)
		# x = tf.constant(divided_image)
		divided_image = tf.convert_to_tensor(divided_image, dtype=tf.float32)
		# print("##################################################### divided_image", x.shape)
		# print("##################################################### divided_image", divided_image[1])
		for i in range(int(divided_image.shape[0])):
			sigle_image = divided_image[i:i+2]
			t1 = time.time()
			# inputs = {}
			# inputs['input_1']=x
			# results = model.predict(np.array(divided_image),batch_size=batch_size)
			outputs = infer(input_1=sigle_image)
			# print("###########################################", outputs['activation_81'].shape)
			t2 = time.time()
			elapsed_time = t2-t1
			print(f"########## 経過時間_batch{batch_size}：{elapsed_time}")
		# y = put_image_together(outputs, (target_img.shape[0], target_img.shape[1]))

		# for label_name, decoded_label in zip(label_info.keys(), decode_prob(binarize_label(y))):
		# 	overlay_img = make_overlay(target_img, decoded_label, label_color='r')
		# 	Thread(target=imageio.imwrite, args=(str(result_path / f"label_{target_img_path.stem}_{label_name}.jpg"), overlay_img), daemon=False).start()


if __name__ == "__main__":
	# main32()
	main16()	

