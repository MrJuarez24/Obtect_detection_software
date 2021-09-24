import cv2


colors_dict = {
    1 : (0,0,0),
    2 : (0,0,50),
    3 : (0,0,75),
    4 : (0,0,100),
    5 : (0,50,0),
    6 : (0,75,0),
    7 : (0,100,0),
    8 : (50,0,0),
}

def print_detections_on_image(object_detection_information, image, trazas, bboxes_format = "coco"):
    """[summary]

    Args:
        object_detection_information ([bboxes, classes, confidences]): Objects detected.
        image (np.ndarray): The image.
        bboxes_format (str) : "coco" is the only format initially avaliable.
    """

    assert bboxes_format == "coco"

    drawn_image = image.copy()

    [bboxes, classes, confidences] = object_detection_information

    for bbox, _class, confidence in zip(bboxes, classes, confidences):
        [x, y, width, height] = bbox
        color = colors_dict[_class] if _class in colors_dict.keys() else (255,0,0)
        if any(_class == x for x in [2,3,4,6,8]):
            drawn_image = cv2.rectangle(drawn_image, (x,y), (x+width, y+height), color, 5)
            drawn_image = cv2.circle(drawn_image, (int(x + (width/2)), int(y + (height/2))), 5, color, -1)
            
            id = ''
            j = 0
            while j < len(trazas):
                if trazas[j][0] == (x + width)/2 and trazas[j][1] == (y + height)/2:
                    id = trazas[j][6]
                j += 1
            
            drawn_image = cv2.putText(drawn_image, str(int(id)), (int(x + (width/2) - 10), int(y)-10) , cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)        #(int(x + (width/2)), int(y + (height/2) + 20))
    return drawn_image