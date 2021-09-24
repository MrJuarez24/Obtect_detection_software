
import numpy as np
import os
from ..object_detector import Object_Detector
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util

class Faster_RCNN_Object_Detector(Object_Detector):
    def __init__(self, model="default"):
        """Constructor.
        Args:
            model (str, optional): if "default", the default pretrained model will be loaded. Else, model should be a path to look for the model.
        """
        super().__init__(model=model)
        if model == "default":
            raise(NotImplementedError)
        else:
            pipeline_path = os.path.join(model, "pipeline.config")
            ckpt_path = os.path.join(model,"checkpoint","ckpt-0")
            print(f"Loading pipeline from {pipeline_path}")
            print(f"Loading checkpoint from {ckpt_path}")
            configs = config_util.get_configs_from_pipeline_file(pipeline_path)
            model_config = configs['model']
            detection_model = model_builder.build(model_config=model_config, is_training=False)

            ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
            ckpt.restore(ckpt_path).expect_partial()

            self.model = detection_model

    def process_output_bbox_format_coco(self, images):
        """Function to apply object detection method and obtain detections with coco structure. 

        Args:
            images (torch.tensor): tf tensor batch of images as shown in Following https://pytorch.org/vision/stable/models.html#faster-r-cnn

        Returns:
            Objects detections with coco structure.
        """

        input_tensor, boxList = self.model.preprocess(images)
        prediction_dict = self.model.predict(input_tensor, boxList)
        detections = self.model.postprocess(prediction_dict, boxList)
        
        outputs = self.turn_faster_rcnn_outputs_into_coco_format(detections)   # We need to manipulate the output.
        return outputs

    
    def preprocess(self, images):
        """ Function to preprocess a batch of images.

        Args:
            images (list(np.ndarray)): batch of images between 0 and 255. Expected dimensions are (batch, height, width, channel) or (batch, channel, height, width).

        Returns:
            A tf tensor with the preprocessed images as self.model expects.
        """  
        assert len(images[0].shape) == 3
        self.tam_images_original = len(images)*[images[0].shape[:2]]
        # We ceate a tensor with dtype float32.
        tf_tensor = tf.convert_to_tensor(np.array(images), dtype=tf.float32)
        # We normalize.

        return tf_tensor

    def turn_faster_rcnn_outputs_into_coco_format(self, detections):
        """Funtion to tf torch faster rcnn output into coco format.

        """
        all_img_bboxes = detections['detection_boxes']
        all_img_classes = detections['detection_classes']
        all_img_scores = detections['detection_scores']
        
        results = []
        for img_bboxes, img_classes, img_scores, img_shape in zip(all_img_bboxes, all_img_classes, all_img_scores, self.tam_images_original):
            new_img_bboxes = []
            img_h, img_w = img_shape
            for bbox in img_bboxes:
                bbox_ymin = int(bbox[0].numpy()*img_h)
                bbox_xmin = int(bbox[1].numpy()*img_w)
                bbox_ymax = int(bbox[2].numpy()*img_h)
                bbox_xmax = int(bbox[3].numpy()*img_w)
                bbox_width = bbox_xmax-bbox_xmin
                bbox_height = bbox_ymax-bbox_ymin
                new_bbox = [bbox_xmin, bbox_ymin, bbox_width, bbox_height]
                new_img_bboxes.append(new_bbox)
            results.append([np.int32(np.array(new_img_bboxes)), np.int32(img_classes)+1, np.float32(img_scores)])
        print("\n---------------------------")
        print(self.tam_images_original)
        print(detections['detection_boxes'])
        print(detections['detection_classes'])
        print(detections['detection_scores'])
        print(results)
        return results
