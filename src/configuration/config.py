"""
config.py is written for define config parameters
"""
import argparse

__author__ = "Amir Mousavi"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"

class BaseConfig:
    """
    This class set static paths and other configs.
    Args:
    argparse :
    The keys that users assign such as sentence, tagging_model and other statictics paths.
    Returns:
    The configuration dict specify text, statics paths and controller flags.
    """
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.run()

    def run(self):
        """
        The run method is written to define config arguments
        :return: None
        """

        self.parser.add_argument("--h5_path",
                                 default="../weights/final_model.h5",
                                 type=str,
                                 help="path of trained model"
                                 )
        self.parser.add_argument("--weight_path",
                                 default="../saved_model",
                                 type=str,
                                 help="path of trained model"
                                 )
        self.parser.add_argument("--target_dir_path",
                                 default="../inference_target/0630/",
                                 type=str,
                                 help="target dir path")
        self.parser.add_argument("--result_path",
                                 default="../inference_result/0630/",
                                 type=str,
                                 help="result path")
        self.parser.add_argument("--train_img_size",
                                 default=(512, 2048),
                                 type=tuple,
                                 help="Total number of training epochs to perform.")
        self.parser.add_argument("--max_workspace_size_bytes",
                                 default=1000000000,
                                 type=int,
                                 help="max workspace size bytes")
        self.parser.add_argument("--output_saved_model_dirFP32",
                                 default="../saved_model_TFTRT_FP32",
                                 type=str,
                                 help="../saved model as TFTRT_FP")
        self.parser.add_argument("--output_saved_model_dirFP16",
                                 default="saved_model_TFTRT_FP16",
                                 type=str,
                                 help="saved model as TFTRT_FP")
        self.parser.add_argument("--saved_model_path",
                                 default="../saved_model/",
                                 type=str,
                                 help="saved model path")
        self.parser.add_argument("--fp32",
                                 default=False,
                                 type=bool,
                                 help="load fp32 model")
        self.parser.add_argument("--fp16",
                                 default=True,
                                 type=bool,
                                 help="load fp16 model")

    def get_args(self):
        """
        The get_args method is written to return config arguments
        :return: argparse
        """
        return self.parser.parse_args()
