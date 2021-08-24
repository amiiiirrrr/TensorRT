
# from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as tftrt
from utils import decode_prob, weighted_crossentropy_wrapper, \
make_overlay, calc_IoU, binarize_label, get_paths_in_dir, \
divide_image, put_image_together, decompose_labels, \
divide_image_borderless, put_image_together_borderless
from tensorflow.keras.models import load_model
# from tensorflow.keras import backend as K
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import copy
import numpy as np
import sys
import time
import imageio
from threading import Thread
from pathlib import Path


class FrozenGraph(object):
	def __init__(self, model, shape):
		shape = (None, shape[0], shape[1], shape[2])
		x_name = 'image_tensor_x'
		tf.compat.v1.disable_eager_execution()
		with K.get_session() as sess:
			x_tensor = tf.compat.v1.placeholder(tf.float32, shape, x_name)
			K.set_learning_phase(0)
			y_tensor = model(x_tensor)
			y_name = y_tensor.name[:-2]
			graph = sess.graph.as_graph_def()
			graph0 = tf.graph_util.convert_variables_to_constants(sess, graph, [y_name])
			graph1 = tf.graph_util.remove_training_nodes(graph0)
		self.x_name = [x_name]
		self.y_name = [y_name]
		self.frozen = graph1

class TfEngine(object):
	"""docstring for TfEngine"""
	def __init__(self, graph):
		g = tf.Graph()
		with g.as_default():
			x_op, y_op = tf.import_graph_def(
				graph_def = graph.frozen,
				return_elements=graph.x_name + graph.y_name
				)
			self.x_tensor = x_op.outputs[0]
			self.y_tensor = y_op.outputs[0]

		config = tf.ConfigProto(
			gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
			allow_growth=True)
			)
		self.sess = tf.Session(graph=g, config=config)

	def infer(self, x):
		y = self.sess.run(
			self.y_tensor,
			feed_dict={self.x_tensor: x}
			)
		return y

class TftrtEngine(TfEngine):
	"""docstring for TftrtEngine"""
	def __init__(self, graph, batch_size, precision):
		tftrt_graph = tftrt.create_inference_graph(
			graph.frozen,
			outputs=graph.y_name,
			max_batch_size=batch_size,
			max_workspace_size_bytes=1 << 30,
			precision_mode=precision,
			minimum_segment_size=2
			)
		opt_graph = copy.deepcopy(graph)
		opt_graph.frozen = tftrt_graph
		super(TftrtEngine, self).__init__(opt_graph)

	def infer(self, x):
		num_tests = x.shape[0]
		y = np.empty((num_tests, self.y_tensor.shape[1]), np.float32)
		batch_size = self.batch_size
		for i in range(0, num_tests, batch_size):
			x_part = x[i : i + batch_size]
			y_part = self.sess.run(self.y_tensor,
						feed_dict={self.x_tensor: x_part})
			y[i : i + batch_size] = y_part
		return y

def main():
	# load pre-trained model
	weight_path = Path("/home/ubuntu/Desktop/o-semantic-segmentation-feature-batch/weights/test2/final_model.h5")
	# inference_targetの中のフォルダ名
	target_name = "0630"
	target_dir_path = Path(__file__).resolve().parent.parent.joinpath("inference_target", target_name)
	assert weight_path.is_file() and target_dir_path.is_dir()
	model: Model = load_model(str(weight_path),
	custom_objects={"weighted_cross_entropy": weighted_crossentropy_wrapper([1., 60., 60.]), 'tf': tf})
	# model = load_model(weight_path)
	model.summary()
	batch_size = 14
	img_shape=(512, 2048, 3)

	result_name = "0630"
	result_path = Path(__file__).resolve().parent.parent.joinpath("inference_result", result_name)

	result_path.mkdir(exist_ok=True)
	image_paths = get_paths_in_dir(target_dir_path)
	image_paths.sort()

	# use Keras to do infer
	# y_keras = model.predict(x_test)
	frozen_graph = FrozenGraph(model, img_shape)
	# tf_engine = TfEngine(frozen_graph)
	# y_tf = tf_engine.infer(x_test)

	tftrt_engine = TftrtEngine(frozen_graph, batch_size, 'FP32')
	# y_tftrt = tftrt_engine.infer(x_test)

	# tftrt_engine = TftrtEngine(frozen_graph, batch_size, 'FP16')
	# y_tftrt = tftrt_engine.infer(x_test)
	for target_img_path in tqdm(image_paths):
		target_img = imageio.imread(str(target_img_path), as_gray=False, pilmode="RGB")
		target_float_img = target_img.astype(np.float32) / 255.
		divided_image =divide_image(target_float_img, train_img_size)
		t1 = time.time()
		results = model.predict(np.array(divided_image),batch_size=batch_size)
		t2 = time.time()
		elapsed_time = t2-t1
		print(f"########## 経過時間_batch{batch_size}：{elapsed_time}")
		y = put_image_together(results, (target_img.shape[0], target_img.shape[1]))

		for label_name, decoded_label in zip(label_info.keys(), decode_prob(binarize_label(y))):
			overlay_img = make_overlay(target_img, decoded_label, label_color='r')
			Thread(target=imageio.imwrite, args=(str(result_path / f"label_{target_img_path.stem}_{label_name}.jpg"), overlay_img), daemon=False).start()

if __name__ == "__main__":
	main()		

		
