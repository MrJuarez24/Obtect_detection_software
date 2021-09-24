import numpy as np
import json

def generate_coco_format_info():
    d = {
        "description": "COCO 2017 Dataset",
        "url": "https://github.com/SirSykon/object_detection",
        "version": "0.1",
        "year": 2021,
        "contributor": "Jorge García <jrggcgz@gmail.com>",
        "date_created": "2021/05/04"
    }

    return d

class Coco_Annotation_Set(object):
    """ 
    Class to contain a set of Coco annotations.
    """
    def __init__(self, json_file_path:str = None, info:dict = None, licenses:list = None, images:list = None, categories:list = None):
        if info is None:
            self.info = generate_coco_format_info()
        else:            
            self.info = info
        if licenses is None:            
            self.licenses = [{"name": "", "id": 0, "url": ""}]
        else:
            self.licenses = licenses
        if images is None:
            self.images = []
        else:
            self.images = images
        if categories is None:
            self.categories = []
        else:
            self.categories = categories         
        
        self.annotations = []
        if not json_file_path is None:
            with open(json_file_path) as json_file:
                data = json.load(json_file)
                for input_ann in data:
                    if input_ann["iscrowd"] == 1:
                        print("We work assuming iscrowd is 0.")
                        raise(NotImplementedError)

                    self.insert_annotation(input_ann["segmentation"], input_ann["area"], input_ann["iscrowd"], input_ann["image_id"], input_ann["bbox"], input_ann["category_id"])


    def insert_image(self, id:int, width:int, height:int, file_name:str = '', _license:int = 0, flickr_url:str = '', coco_url:str = '', date_captured:int = 0):
        """Method to insert image into Coco Annotation Structure.

        Args:
            id (int): Image identifier.
            width (int): Image Width.
            height (int): Image Height.
            file_name (str, optional): Image file name.. Defaults to ''.
            _license (int, optional): License id. Defaults to 0.
            flickr_url (str, optional): Image flickr URL. Defaults to ''.
            coco_url (str, optional): Image COCO URL. Defaults to ''.
            date_captured (int, optional): Image date. Defaults to 0.
        """

        img = {
            'id': id,
            'width': width,
            'height': height,
            'file_name': file_name,
            'license' : _license,
            'flickr_url' : flickr_url,
            'coco_url' : coco_url,
            'date_captured' : date_captured
        }

        self.images.append(img)

    def insert_coco_annotation_object(self, coco_annotation_object):
        self.annotations.append(coco_annotation_object)

    def insert_annotation(self, id, segmentation, area, iscrowd, image_id, bbox, category_id):
        ann = Coco_Annotation_Object(
            id = id,
            segmentation = segmentation,
            area = area,
            iscrowd = iscrowd,
            image_id = image_id,
            bbox = bbox,
            category_id = category_id
            )
        self.annotations.append(ann)

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self, only_essential_data = True):

        anns = []
        for ann in self.annotations:
            anns.append(ann.to_dict(only_essential_data = only_essential_data))

        d = {
            "info" : self.info,
            "licenses" : self.licenses,
            "categories" : self.categories,
            "images" : self.images,
            "annotations" : anns
        }

        return d

    def annotations_list_dict(self):
        return self.to_dict()["annotations"]

    def __str__(self):
        return str(self.to_dict())

    def to_json(self, save_path, only_annotations = True, only_essential_data = True):
        
        data = self.to_dict(only_essential_data=only_essential_data)

        if only_annotations: 
            data = data["annotations"]
            
        with open(save_path, "w") as f:
            json.dump(data, f)


class Coco_Annotation_Object(object):

    """
    Class to contain Image annotations information following COCO format as described in https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    Since initially this repository is to work with object detection methods, some attributes could be ignored.

    Example of annotations list: 

    The following JSON shows 2 different annotations.

    The first annotation:

        Has a segmentation list of vertices (x, y pixel positions)

        Has an area of 702 pixels (pretty small) and a bounding box of [473.07,395.93,38.65,28.67]

        Is not a crowd (meaning it’s a single object)

        Is category id of 18 (which is a dog)

        Corresponds with an image with id 289343 (which is a person on a strange bicycle and a tiny dog)

    The second annotation:

        Has a Run-Length-Encoding style segmentation

        Has an area of 220834 pixels (much larger) and a bounding box of [0,34,639,388]

        Is a crowd (meaning it’s a group of objects)

        Is a category id of 1 (which is a person)

        Corresponds with an image with id 250282 (which is a vintage class photo of about 50 school children)

    "annotations": [
        {
            "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],
            "area": 702.1057499999998,
            "iscrowd": 0,
            "image_id": 289343,
            "bbox": [473.07,395.93,38.65,28.67],
            "category_id": 18,
            "id": 1768
        },
        ...
        {
            "segmentation": {
                "counts": [179,27,392,41,…,55,20],
                "size": [426,640]
            },
            "area": 220834,
            "iscrowd": 1,
            "image_id": 250282,
            "bbox": [0,34,639,388],
            "category_id": 1,
            "id": 900100250282
        }
    ]

    Args:
        bbox (list): defines the position of the object. its structure changes according to bbox_format as follows:
            "coco" -> [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
            "absolute" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2]
            "relative" -> []. Defaults to None.
        category_id (int): Cateogry identification number. Defaults to None.
        id (int, optional): Object identification number. Defaults to None.
        segmentation (list, optional): Semantic segmentation vertices. Defaults to None.
        area (float, optional): Area accupied by the object. Defaults to None.
        iscrowd (int, optional): ¿Is crowd? 1 if True, 0 otherwise. Currently assumed 0. Defaults to 0.
        image_id (int, optional): Image identification number. Defaults to None.
        image_shape (list, optional): Image object shape. Defaults to None.
        bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".

    """

    def __init__(self, bbox:list, category_id:int, score:float, image_id:int, id:int = None, segmentation:list = [], area:float = None, iscrowd:int = 0, image_shape:list = None, bbox_format:str = "coco"):
        """
        Args:
            bbox (list): defines the position of the object. its structure changes according to bbox_format as follows:
                "coco" -> [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
                "absolute" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2]
                "relative" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1 * self.shape[1],y1 * self.shape[0]], [x2 * self.shape[1], y1 * self.shape[0]], [x1 * self.shape[1],y2 * self.shape[0]], [x2 * self.shape[1], y2 * self.shape[0]]. 
                Defaults to None.
            category_id (int): Cateogry identification number. Defaults to None.
            id (int, optional): Object identification number. Defaults to None.
            segmentation (list, optional): Semantic segmentation vertices. Defaults to None.
            area (float, optional): Area accupied by the object. Defaults to None.
            iscrowd (int, optional): ¿Is crowd? 1 if True, 0 otherwise. Currently assumed 0. Defaults to 0.
            image_id (int, optional): Image identification number. Defaults to None.
            image_shape (list, optional): Image object shape. Defaults to None.
            bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".
            score (float): defines the category_id confidence.

        Returns:
            Coco_Annotation_Object: Class to contain Image annotations information following COCO format as described in https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch 
            Since initially this repository is to work with object detection methods, some attributes could be ignored.
        """
        self.segmentation = np.array(segmentation)
        self.area = area
        if iscrowd == 1:
            print("WARNING: Current Coco_Format_Annotation class asusumes iscrowd is 0 (False).")
            self.iscrowd = True
        else:
            self.iscrowd = False
        self.image_id = image_id
        self.bbox = np.array(bbox)
        self.category_id = category_id
        self.score = score
        self.id = id

        self.bbox_format = bbox_format
        self.image_shape = image_shape

        if np.max(self.bbox) < 1:
            "If there is no"

    def __str__(self):
        return str(self.to_dict())

    def set_image_shape(self, input_shape:list) -> None:
        """Method to set image shape where this object appears.

        Args:
            input_shape (list[int]): Shape.

        Returns:
            None
        """
        self.image_shape = input_shape

    def to_dict(self, only_essential_data = True) -> dict:
        """Method to turn this annotation into a dictionary with only coco data.

        Returns:
            dict[str, object]: dictionary with only coco data.
        """
        d = {
            "image_id" : int(self.image_id),
            "bbox" : self.bbox.tolist(),
            "category_id" : int(self.category_id),
            "score" : float(self.score)
            }
        if not only_essential_data:
            d["id"] = int(self.id)
            d["segmentation"] = self.segmentation.tolist()
            d["area"] =  self.area
            d["iscrowd"] = self.iscrowd

        return d

    def get_corners_relative_format(self) -> list:
        """
        Returns:
            list: [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1 * self.shape[1],y1 * self.shape[0]], [x2 * self.shape[1], y1 * self.shape[0]], [x1 * self.shape[1],y2 * self.shape[0]], [x2 * self.shape[1], y2 * self.shape[0]]. Defaults to None.
        """

        if self.bbox_format == "coco":
            assert self.shape

            x1, y1, width, height = self.bbox
            return [x1 / self.shape[1],y1 / self.shape[0], (x1 + width) / self.shape[1], (y1 - height) / self.shape[0]]
        
        if self.bbox_format == "relative":
            return self.bbox_format.tolist()

        if self.bbox_format == "absolute":
            assert self.shape

            x1, y1, x2, y2 = self.bbox
            bbox = [x1 / self.shape[1],y1 / self.shape[0], x2 / self.shape[1], y2 / self.shape[0]]

    def get_corners_absolute_format(self) -> list:
        """
        Returns:
            list: [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2].
        """
        
        if self.bbox_format == "coco":
            x1, y1, width, height = self.bbox
            return [x1, y1, x1+width, y1+height]
        
        if self.bbox_format == "relative":
            assert self.shape

            x1, y1, x2, y2 = self.bbox
            bbox = [x1 * self.shape[1],y1 * self.shape[0], x2 * self.shape[1], y2 * self.shape[0]]

        if self.bbox_format == "absolute":
            return self.bbox_format.tolist()

    def get_corners_coco_format(self) -> list:
        """
        Returns:
            list: [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
        """
        
        if self.bbox_format == "coco":
            return self.bbox_format.tolist()
        
        if self.bbox_format == "relative":

            x1, y1, x2, y2 = self.get_corners_absolute_format()
            bbox = [x1, y1, x2-x1, y2-y1]

        if self.bbox_format == "absolute":
            x1, y1, x2, y2 = self.bbox
            return [x1, y1, x2-x1, y2-y1]

    def get_bbox_corners(self, format:str = "coco") -> list:
        """Method to get corners.

        Args:
            format (str, optional): format defines the corners format. Defaults to "coco".

        Returns:
            list :
                It dependes on format: 
                "coco" -> [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
                "absolute" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2]
                "relative" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1 * self.shape[1],y1 * self.shape[0]], [x2 * self.shape[1], y1 * self.shape[0]], [x1 * self.shape[1],y2 * self.shape[0]], [x2 * self.shape[1], y2 * self.shape[0]]. Defaults to None.
        """

        if format == "coco":
            return self.get_corners_coco_format()
        
        if format == "relative":
            return self.get_corners_relative_format()

        if format == "absolute":
            return self.get_corners_absolute_format()

def new_object(bbox:list, category_id:int, id:int = None, segmentation:list = None, area:float = None, iscrowd:int = 0, image_id:int = None, image_shape:list = None, bbox_format:str = "coco") -> Coco_Annotation_Object:
    """
    Args:
        bbox (list): defines the position of the object. its structure changes according to bbox_format as follows:
            "coco" -> [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
            "absolute" -> [x1,y1,x2,y2] with corners (left-right, top-bottom) [x1,y1], [x2, y1], [x1,y2], [x2, y2]
            "relative" -> []. Defaults to None.
        category_id (int): Cateogry identification number. Defaults to None.
        id (int, optional): Object identification number. Defaults to None.
        segmentation (list, optional): Semantic segmentation vertices. Defaults to None.
        area (float, optional): Area accupied by the object. Defaults to None.
        iscrowd (int, optional): ¿Is crowd? 1 if True, 0 otherwise. Currently assumed 0. Defaults to 0.
        image_id (int, optional): Image identification number. Defaults to None.
        image_shape (list, optional): Image object shape. Defaults to None.
        bbox_format (str, optional): defines how bbox is. Format can be "coco" (default), "absolute" or "relative". Defaults to "coco".

    Returns:
        Coco_Annotation_Object: Class to contain Image annotations information following COCO format as described in https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch 
        Since initially this repository is to work with object detection methods, some attributes could be ignored.
    """

    return Coco_Annotation_Object(bbox, category_id, id=id, segmentation=segmentation, area=area, iscrowd=iscrowd, image_id = image_id, image_shape = image_shape, bbox_format = bbox_format)

def object_from_dict(self, input_dict:dict) -> Coco_Annotation_Object:
    """Method to generate a coco format annotation from a dictionary.

    Args:
        input_dict (dict[str, object]): dict with coco annotation format data.

    Returns:
        Coco_Annotation_Object: Class to contain Image annotations information following COCO format as described in https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch 
        Since initially this repository is to work with object detection methods, some attributes could be ignored.
    """

    return new_object(
        np.array(input_dict["bbox"]),
        input_dict["category_id"],
        id = input_dict["id"],
        segmentation = np.array(input_dict["segmentation"]), 
        area = input_dict["area"], 
        iscrowd = input_dict["iscrowd"],
        image_id = input_dict["image_id"]
    )

def read_annotations_from_json_file(file_path):
    """

    Args:
        file_path (str): Path to json file containing the annotations.
    """

    with open(file_path) as json_file:
        data = json.load(json_file)
        print(data.keys())
        print(data['images'][0])
        print(data['annotations'][0])
