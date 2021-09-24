from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pylab
import json 


def evaluate(gt_annotation_file_path, res_annotation_file_path):
    """[summary]

    Args:
        gt_annotation_file_path (str): Path to groundtruth .json annotations file.
        res_annotation_file_path (str): Path to results .json annotations file.
    """

    pylab.rcParams['figure.figsize'] = (10.0, 8.0)


    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    print ('Running demo for %s results.'%(annType))

    # use the valadation labelme file
    cocoGt=COCO(gt_annotation_file_path)
    #initialize COCO detections api
    # use the generated results
    cocoDt=cocoGt.loadRes(res_annotation_file_path)


    dts = json.load(open(res_annotation_file_path,'r'))
    imgIds = [imid['image_id'] for imid in dts]
    imgIds = sorted(list(set(imgIds)))

    '''
    imgIds=sorted(cocoGt.getImgIds())
    imgIds=imgIds[0:24]
    imgId = imgIds[np.random.randint(24)]
    '''

    # running box evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)

    cocoEval.params.imgIds  = imgIds

    #cocoEval.params.catIds = [3] # 1 stands for the 'person' class, you can increase or decrease the category as needed

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()