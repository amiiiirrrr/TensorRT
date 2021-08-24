"""
run_tftrt.py is written for quantizing model
"""

from threading import Thread
from tqdm import tqdm
import time
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
					make_overlay, binarize_label, \
					get_paths_in_dir, put_image_together
import collections
from pathlib import Path
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from configuration import BaseConfig
from tools import preprocess_image

__author__ = "Amir Mousavi"
__project__ = "TF-TRT"
__credits__ = ["Amir Mousavi"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"

class TfTrt:
	def __init__(self, args):
		self.args = args
		weight_path = Path(self.args.h5_path)
		target_dir_path = Path(self.args.target_dir_path)
		result_path = Path(self.args.result_path)
		assert weight_path.is_file() and target_dir_path.is_dir()
		self.h5_model: Model = load_model(str(weight_path),
		custom_objects={"weighted_cross_entropy": 
						weighted_crossentropy_wrapper([1., 60., 60.]), 'tf': tf})

		result_path.mkdir(exist_ok=True)
		self.image_paths = get_paths_in_dir(target_dir_path)
		self.image_paths.sort()

		self.h5_model.save(self.args.saved_model_path)

	def convertToFp32(self):
		"""
		The convertToFp32 method is convert saved model to FP32
		:return: None
		"""

		print('Converting to TF-TRT FP32...')
		conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
		precision_mode=trt.TrtPrecisionMode.FP32,
		max_workspace_size_bytes=self.args.max_workspace_size_bytes)

		converter = trt.TrtGraphConverterV2(
		input_saved_model_dir=str(self.args.saved_model_path),
		conversion_params=conversion_params)

		converter.convert()
		converter.save(output_saved_model_dir=self.args.output_saved_model_dirFP32)
		print('Done Converting to TF-TRT FP32')

	def convertToFp16(self):
		"""
		The convertToFp16 method is convert saved model to FP16
		:return: None
		"""

		print('Converting to TF-TRT FP16...')
		conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
		precision_mode=trt.TrtPrecisionMode.FP16,
		max_workspace_size_bytes=self.args.max_workspace_size_bytes)

		converter = trt.TrtGraphConverterV2(
		input_saved_model_dir=str(self.args.saved_model_path),
		conversion_params=conversion_params)

		converter.convert()
		converter.save(output_saved_model_dir=self.args.output_saved_model_dirFP16)
		print('Done Converting to TF-TRT FP16')

	def load_infer_model(self):
		"""
		The load_infer_model method is written to load FP16 model  
		:return: infer model
		"""

		if self.args.fp32:
			self.convertToFp32()
			saved_model_loaded = tf.saved_model.load(
				self.args.output_saved_model_dirFP32, tags=[tag_constants.SERVING])
		if self.args.fp16:
			self.convertToFp16()
			saved_model_loaded = tf.saved_model.load(
			self.args.output_saved_model_dirFP16, tags=[tag_constants.SERVING])

		signature_keys = list(saved_model_loaded.signatures.keys())
		# print(signature_keys)

		infer = saved_model_loaded.signatures['serving_default']
		# print(infer.structured_outputs)
		# print(infer)
		return infer

	def run(self):
		"""
		The run method is written to execute model  
		:return: None
		"""

		label_info = collections.OrderedDict()
		label_info["background"] = (0, 0, 0)
		label_info["kizu"] = (128, 0, 0)

		infer = self.load_infer_model()

		for target_img_path in tqdm(self.image_paths):
			divided_image = preprocess_image(target_img_path, self.args.train_img_size)
			t1 = time.time()
			for i in range(int(divided_image.shape[0])):

				# print("########### divided_image[i] ############", divided_image[i].shape)
				sigle_image = tf.expand_dims(divided_image[i], 0)
				# sigle_image = divided_image[i:i+2]
				# print("############### sigle_image[i] #############", sigle_image.shape)
				outputs = infer(input_1=sigle_image)
				t2 = time.time()
				output_single_image = outputs['activation_81']
			elapsed_time = t2-t1
			print(f"########## 経過時間：{elapsed_time}")
				# print("############### output_single_image[i] #############", output_single_image.shape)
			# y = put_image_together(outputs, (target_img.shape[0], target_img.shape[1]))
			# for label_name, decoded_label in zip(label_info.keys(), decode_prob(binarize_label(y))):
			# 	overlay_img = make_overlay(target_img, decoded_label, label_color='r')
			# 	Thread(target=imageio.imwrite, args=(str(result_path / f"label_{target_img_path.stem}_{label_name}.jpg"), overlay_img), daemon=False).start()


if __name__ == "__main__":
	
	args = BaseConfig().get_args()
	obj_TfTrt = TfTrt(args)
	obj_TfTrt.run()
		


