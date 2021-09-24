"""
Author: Jorge García <sirsykon@gmail.com>
"""

import numpy as np

class Object_Detector():
    """
    Class to act as abstract class in order to create wrappers following the same structure.

        Args:
            bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
    """
    def __init__(self, bbox_format="coco", model="default"):
        """
        Method to initialize the network.

        Args:
            bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
            model (str, optional): if "default", the default pretrained model will be loaded. Else, model should be a path to look for the model.
        """
        self.set_bbox_format(bbox_format)
        self.model_origin = model

    def set_bbox_format(self, bbox_format):
        """
        Args:
            bbox_format (str): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
        """
        self.bbox_format = bbox_format

    def process_output_bbox_format_coco(self, images):
        """ Abstract function to apply object detection method and obtain detections with coco structure. 
            Any child class must implement this function.

        Args:
            images (list(np.ndarray)): List of images.

        Returns:
            The output is a list containing the object detection information for each image as follows:
                output : list(objec_detection_information)
                objec_detection_information : [bboxes, classes, confidences]
                bboxes : list(bbox) defines the positions of the objects. bbox structure is as folows:
                    [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
                classes : list(int) 
                confidences: list(float)
        """
        raise(NotImplementedError)

    def preprocess(self, images):
        """Method to preprocess images in order to be processed by this object detector.
            Any child class must implement this function.
            
        Args:
            images (list(np.ndarray)): input images to be preprocessed. 
        """
        raise(NotImplementedError)

    def filter_output_by_confidence_treshold(self, objects:list, treshold:float = 0.5):
        """Function to filter the detections with treshold.
        Args:
            objects (list): List of object detections.
            treshold (float, optional): Trshold to filter detections with. Defaults to 0.5.
        """
        filtered_objects = []

        for obj in objects:
            bboxes, classes, confidences = obj
            idx = confidences >= treshold
            filtered_objects.append([bboxes[idx], classes[idx], confidences[idx]])

        return filtered_objects

    def process(self, images):
        """Method to get object detection information.

        Args:
            images : list(np.ndarray): List of images.

        Returns:
            The output is a list containing the object detection information for each image as follows:
                output : list(objec_detection_information)
                objec_detection_information : [bboxes, classes, confidences]
                bboxes : list(bbox) defines the positions of the objects. bbox structure changes according to bbox_format as follows:
                    "coco" -> [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
                    "absolute" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2]
                    "relative" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1 * self.shape[1],y1 * self.shape[0]], [x2 * self.shape[1], y1 * self.shape[0]], [x1 * self.shape[1],y2 * self.shape[0]], [x2 * self.shape[1], y2 * self.shape[0]]. 
                classes : list(int) 
                confidences : list(float)
        """
        output = self.process_output_bbox_format_coco(images)
        if not self.bbox_format == "coco":
            new_output = []
            for [coco_format_bboxes, classes, confidences] in output:
                new_format_bboxes = []
                for coco_format_bbox in coco_format_bboxes:
                    x1, y1, width, height = coco_format_bbox
                    if self.bbox_format == "relative":

                        assert self.shape
                        relative_format_bbox = [x1 / self.shape[1],y1 / self.shape[0], (x1 + width) / self.shape[1], (y1 - height) / self.shape[0]]
                        new_format_bboxes.append(relative_format_bbox)

                    if self.bbox_format == "absolute":

                        absolute_format_bbox = [x1, y1, x1+width, y1+height]
                        new_format_bboxes.append(relative_format_bbox)

                new_output.append([new_format_bboxes, classes, confidences])

        else:
            return output

    def __call__(self, images):
        return self.process(images)

class Dummy_Object_Detector(Object_Detector):
    """Class to implement Object_Detector wrapper as an example dummy Object_Detector.
    """

    def __init__(self):
        super().__init__()
        print("Creating Dummy Object Detector")

    def preprocess(self,images):
        """[summary]

        Args:
            images ([type]): [description]

        Returns:
            [type]: [description]
        """
        return images


    

