
"""
Author: Emilio Benzo <emiliojbenzo@gmail.com>
"""
import cv2
import torch
import numpy as np
from ..object_detector import Object_Detector


class YOLO_Object_Detector(Object_Detector):

    def __init__(self, to="cuda", model="default"):
        """
        Method to initialize the network.

        Args:
            bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
            model (str, optional): if "default", the default pretrained model will be loaded. Else, model should be a path to look for the model.
        """
        super().__init__()
        if model=="default":
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
        else:
            raise(NotImplementedError)
        self.model.to(to)
        self.to = to
        self.tam_images_original = []



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
        results = self.model(images)
        r_img_cpu_list=[]
        for r_img in results.xyxy:
            r_img_cpu = r_img.to("cpu").numpy()
            r_img_cpu_list.append(r_img_cpu)
        outputs = self.turn_YOLO_rcnn_outputs_into_coco_format(r_img_cpu_list)
                       
        return outputs


    def preprocess(self, images):
        """Method to preprocess images in order to be processed by this object detector.
        Args:
        images (list(np.ndarray)): input images to be preprocessed. 
        
        #print("entro en el preprocess")
        self.tam_images_original = []
        preprocess_images = []
        for im_rgb in images: 
            self.tam_images_original.append(im_rgb.shape)#tupla con la estructura de la imagen
            #im_rgb = cv2.resize(im_rgb, (416,416))#tamaño maximo de la imagen
            preprocess_images.append(im_rgb)#creamos una lista 
        tensor = torch.as_tensor(preprocess_images)
        print(tensor)
        """
        return images

    def turn_YOLO_rcnn_outputs_into_coco_format(self, images):
        new_results = []
        for process_img_results in images:
            new_bboxes = []
            new_classes = []
            new_confidences = []
            for object_each_img in process_img_results:
                xmin, ymin, xmax, ymax, confidence, class_ = object_each_img
                w = xmax - xmin
                h = ymax - ymin
                new_bbox  = [xmin, ymin, w, h]
                new_bboxes.append(new_bbox)
                new_classes.append(class_)
                new_confidences.append(confidence)
            array_bboxes = np.int32(np.array(new_bboxes))
            array_classes = np.int32(np.array(new_classes)) + 1 # Coco uses 0 as "background", not as "person"
            array_confidences = np.array(new_confidences)
            new_results.append([array_bboxes, array_classes, array_confidences])  
        
        return new_results


