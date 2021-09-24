import sys
import os
import coco_format_utils

class Config():
    
    
    def __init__(self):
        # Tensorflow object detection root directory.
        self.TENSORFLOW_RESEARCH_ROOT_FOLDER = os.path.abspath("../../Tensorflow/models/research")
        # Tensorflow object detection models folder
        self.TENSORFLOW_DOWNLOADED_MODELS_FOLDER = os.path.abspath(f"../../Tensorflow/downloaded_models")
        
        # Model Name to use.
        self.MODEL_NAME = "ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8"
        
        # Image shape.
        self.SHAPE = [1024,1024]
        
        # Paths to model information.
        self.PIPELINE_CONFIG = os.path.abspath(f"{self.TENSORFLOW_RESEARCH_ROOT_FOLDER}/object_detection/configs/tf2/{self.MODEL_NAME}.config")
        self.CHECKPOINT_PATH = os.path.abspath(f"{self.TENSORFLOW_DOWNLOADED_MODELS_FOLDER}/{self.MODEL_NAME}/checkpoint/ckpt-0")
        
        self.INPUT_FOLDER = "../input"
        self.OUTPUT_FOLDER = f"../output/test/{self.MODEL_NAME}"
        
        # Training hyper-parameters
        self.BATCH_SIZE = 4
        self.LEARNING_RATE = 0.001
        self.EPOCHS = 2
        self.TEST_SPLIT = 0.3
        self.VALIDATION_SPLIT = 0.1
        
        sys.path.append(self.TENSORFLOW_RESEARCH_ROOT_FOLDER)

        coco_format_utils.n
