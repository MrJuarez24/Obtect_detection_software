import torch
import torchvision
import numpy as np
from ..object_detector import Object_Detector

# Following https://pytorch.org/vision/stable/models.html#faster-r-cnn

class Faster_RCNN_Object_Detector(Object_Detector):
    def __init__(self, to="cuda", model="default"):
        """Constructor.
        Args:
            to (str, optional): "cuda" (GPU) or "cpu" selects where to process the information. Defaults to "cuda".
            model (str, optional): if "default", the default pretrained model will be loaded. Else, model should be a path to look for the model.
        """
        super().__init__(model=model)
        if model == "default":
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.eval()
        else:
            raise(NotImplementedError)
        self.model.to(to)
        self.to = to

    def process_output_bbox_format_coco(self, images):
        """Function to apply object detection method and obtain detections with coco structure. 

        Args:
            images (torch.tensor): torch tensor batch of images as shown in Following https://pytorch.org/vision/stable/models.html#faster-r-cnn

        Returns:
            Objects detections with coco structure.
        """
        # We ensure images is a torch tensor with channel-first.
        assert type(images) == torch.Tensor
        assert list(images.size())[1] == 3

        with torch.no_grad():   # We are not training so we do not need grads.
            outputs = self.model(images)

        outputs = self.turn_faster_rcnn_outputs_into_coco_format(outputs)   # We need to manipulate the output.
        return outputs
    
    def preprocess(self, images, to = "cuda"):
        """ Function to preprocess a batch of images.

        Args:
            images (list(np.ndarray)): batch of images between 0 and 255. Expected dimensions are (batch, height, width, channel) or (batch, channel, height, width).

        Returns:
            A torch tensor with the preprocessed images as self.model expects.
        """  
        assert len(images[0].shape) == 3

        # We copy the images np array to ensure they are not destroyed.
        imgs = images.copy()
        if imgs[0].shape[-1] == 3: # We guess the last dimensions is channels.                
            # We need channel first so we transform the data.
            imgs = np.moveaxis(imgs,-1, 1)
        # We normalize.
        imgs = imgs/255.
        # We ceate a tensor with dtype float32.
        return torch.tensor(imgs, dtype=torch.float32).to(device=self.to)

    def turn_faster_rcnn_outputs_into_coco_format(self, all_images_output):
        """Funtion to turn torch faster rcnn output into coco format.

        Args:
            all_images_output (dictionary): A dictionary with the following structure:
                {"boxes" : list(box),
                labels : list(int),
                scores : list(float)}
            with  box = [x1,y1,x2,y2] for each object with (x1,y1) as topp-left corner and (x2,y2) with bottom-right corner-

        Returns:

        """
        new_all_images_output = []
        for img_output in all_images_output:    # For each image objects detections...
            # We get the information.
            bboxes = img_output["boxes"]
            labels = img_output["labels"]
            scores = img_output["scores"]
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0]
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

            if self.to == "cuda":
                bboxes = bboxes.to("cpu")
                labels = labels.to("cpu")
                scores = scores.to("cpu")
            
            bboxes = np.int32(bboxes.numpy())
            labels = np.int32(labels.numpy())
            scores = scores.numpy()
            new_all_images_output.append([bboxes, labels, scores])
        
        return new_all_images_output
