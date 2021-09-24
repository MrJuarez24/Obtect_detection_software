"""
Author: Emilio Benzo <emiliojbenzo@gmail.com>
"""
from ..object_detector import Object_Detector
import cv2
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

class SSD_Object_Detector(Object_Detector):

    def __init__(self, to="cuda", model="default"):
        """
        Class to act as abstract class in order to create wrappers following the same structure.

            Args:
                bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
                model (str, optional): if "default", the default pretrained model will be loaded. Else, model should be a path to look for the model.
        """
        super().__init__(model=model)
        if model == "default":        
            self.precision = 'fp32'
            self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=self.precision)
            self.model.eval()
        else:
            raise(NotImplementedError)
        self.model.to(to)
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        self.to = to
        self.tam_images_original = []


        #"""
        #Method to initialize the network.

        #Args:
        #    bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
        #"""
        #self.set_bbox_format(bbox_format)

    

    def process_output_bbox_format_coco(self, images):
        """[summary]

        Args:
            images (list(np.ndarray)): List of images.

        Returns:
            The output is a list containing the object detection information for each image as follows:
                output : list(objec_detection_information)
                objec_detection_information : [bboxes, classes, confidences]
                bboxes defines the position of the object. its structure changes according to bbox_format as follows:
                    [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
                classes : list(int) 
                confidences: list(float)
        """
        with torch.no_grad():   # We are not training so we do not need grads.
            detections_batch = self.model(images)
        results_per_input = self.utils.decode_results(detections_batch)
        outputs = self.turn_ssd_rcnn_outputs_into_coco_format(results_per_input)

        return outputs

    def preprocess(self, images):
        """Method to preprocess images in order to be processed by this object detector.

        Args:
        images (list(np.ndarray)): input images to be preprocessed. 
        """
        self.tam_images_original = []
        preprocess_images = []
        for im_rgb in images: 
            self.tam_images_original.append(im_rgb.shape)#tupla con la estructura de la imagen
            im_rgb = cv2.resize(im_rgb, (300,300))
            im_rgb = (im_rgb - 127.5)/127.5
            preprocess_images.append(im_rgb)
        tensor = self.utils.prepare_tensor(preprocess_images, self.precision == 'fp16')
        #++ tensor.contiguous()#añadido para arreglar error de memoria
        #The model requires input tensor to be contiguous in memory. The util function (and the tutorial) missed that, but I made MR with a fix.
        return tensor.to(self.to)


    def turn_ssd_rcnn_outputs_into_coco_format(self, results):

        #print(results)
        new_results = []
        for res, original_shape in zip(results, self.tam_images_original):
            bboxes, classes, confidences = res
            height, width, _ = original_shape
            new_bboxes = []
            for bbox in bboxes:
                left, top, right, bot = bbox
                #x, y, w, h = [val * 300 for val in [left, top, right - left, bot - top]]
                x, w = [val * width for val in [left, right-left]]
                y, h = [val * height for val in [top, bot-top]]
                new_bbox = [x, y, w, h]
                new_bboxes.append(new_bbox)
            new_results.append([np.int32(np.array(new_bboxes)), np.int32(classes), confidences])

        return new_results

