import numpy as np

def intersection_between_two_rectangles(sq1, sq2):
    """
    Function to calculate the intersection area between two rectangle. Squares are assumed to be given as list: [min_height, min_width, max_height, max_width]

    Args:
        sq1 ([type]): [description]
        sq2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    sq1_min_h = sq1[0]
    sq1_max_h = sq1[2]
    sq1_min_w = sq1[1]
    sq1_max_w = sq1[3]
    
    sq2_min_h = sq2[0]
    sq2_max_h = sq2[2]
    sq2_min_w = sq2[1]
    sq2_max_w = sq2[3]
    
    # We will create the intersection rectangle.
    in_sq_min_h = max(sq1_min_h, sq2_min_h)
    in_sq_max_h = min(sq1_max_h, sq2_max_h)
    in_sq_min_w = max(sq1_min_w, sq2_min_w)
    in_sq_max_w = min(sq1_max_w, sq2_max_w)
    
    area = rectangle_area([in_sq_min_h, in_sq_min_w, in_sq_max_h, in_sq_max_w])
    
    if area != -1:
        return area
    else:
        return 0

def intersection_between_two_masks(mask1, mask2):
    """
    Function to calculate the intersection area between to masks. Masks should have the same shape and are assumed to be boolean or binary.

    Args:
        mask1 ([type]): [description]
        mask2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    assert mask1.shape==mask2.shape

    intersection_mask = np.logical_and(mask1, mask2)*1

    return np.sum(intersection_mask)

def union_between_two_masks(mask1, mask2):
    """
    Function to calculate the union area between to masks. Masks should have the same shape and are assumed to be boolean or binary.

    Args:
        mask1 ([type]): [description]
        mask2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    assert mask1.shape==mask2.shape
    
    union_mask = np.logical_or(mask1, mask2)*1

    return np.sum(union_mask)
 
def union_of_two_rectangles(sq1, sq2):
    """
    Function to calculate the union area of two rectangle. Squares are assumed to be given as list: [min_height, min_width, max_height, max_width]   

    Args:
        sq1 ([type]): [description]
        sq2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    sq1_area = rectangle_area(sq1)
    sq2_area = rectangle_area(sq2)
    
    return sq1_area + sq2_area - intersection_between_two_rectangles(sq1, sq2)    

def rectangle_area(sq):
    """
    Function to calculate the area of a rectangle. Square is assomed to be given as a list: [min_height, min_width, max_height, max_width]

    Args:
        sq ([type]): [description]

    Returns:
        [type]: [description]
    """
    sq_min_h = sq[0]
    sq_max_h = sq[2]
    sq_min_w = sq[1]
    sq_max_w = sq[3]
    
    height = sq_max_h - sq_min_h
    
    if height < 0:
        return -1
    
    width = sq_max_w - sq_min_w
    
    if width < 0:
        return -1
    
    return height*width
  
def intersection_over_union_using_squares(sq1, sq2):
    """
    Function to calculate the intersection over union. Squares are assomed to be given as a list: [min_height, min_width, max_height, max_width] 

    Args:
        sq1 ([type]): [description]
        sq2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    intersection = intersection_between_two_rectangles(sq1,sq2)
    union = union_of_two_rectangles(sq1,sq2)
    
    return intersection/union

def intersection_over_union_using_masks(mask1, mask2):

    intersection = intersection_between_two_masks(mask1, mask2)
    union = union_between_two_masks(mask1, mask2)

    return intersection/union