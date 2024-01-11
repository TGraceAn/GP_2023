import numpy as np
import cv2

"""
This is use for drawing the bounding boxes and the labels on the image
Should be able to implement this in the main.py file for the final output
"""


class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# class_names_2 = ['bed', 'night_stand', 'ottoman', 'dresser', 'lamp', 'pillow',
#                 'mirror', 'chair', 'sofa', 'monitor', 'cabinet', 'sofa_chair',
#                 'table', 'computer', 'door', 'tv', 'box', 'bottle', 'book',
#                 'coffee_table', 'laptop', 'shelf', 'plant', 'desk', 'endtable',
#                 'fridge', 'recycle_bin', 'garbage_bin', 'bench', 'bookshelf',
#                 'printer', 'counter', 'toilet', 'sink', 'towel', 'vanity',
#                 'painting', 'drawer', 'keyboard', 'paper', 'books', 'whiteboard',
#                 'picture', 'cpu', 'stool', 'curtain', 'cloth', 'person', 'stair']

class_names_3 =['bed', 'night_stand', 'ottoman', 'dresser', 'lamp', 'pillow', 'mirror', 'chair', 'sofa','monitor','cabinet',
                'table', 'computer', 'door', 'tv', 'box', 'bottle', 'book', 'laptop', 'shelf', 'plant', 'desk', 'fridge',
                'recycle_bin', 'garbage_bin', 'bench', 'bookshelf', 'printer', 'counter', 'toilet', 'sink', 'towel',
                'vanity', 'painting', 'drawer', 'keyboard', 'paper', 'whiteboard', 'picture', 'cpu', 'stool', 'curtain','cloth',
                'person', 'stair']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names_3), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

#    Draw masks
#     det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections

    for class_id, box, score in zip(class_ids, boxes, scores):

        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names_3[class_id]

        caption = f'{label} {int(score * 100)}%'
        
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple = (0, 0, 255),
thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple = (255, 0, 0),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 + th), color, -1)

    return cv2.putText(image, text, (x1, y1 + th), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), text_thickness, cv2.LINE_AA)


# USELESS FOR NOW

def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

def final_object_dict(class_ids):
    final_list = []
    for class_id in class_ids:
        label = class_names_3[class_id]
        final_list.append(label)
    
    return final_list
    
def cal_warning_depth(depth_map):
    depth_top_center = depth_map[0:160, 266:373]
    depth_mid_center = depth_map[160:320, 266:373]
    depth_bottom_center = depth_map[320:480, 266:373]
    depth_value_top = np.mean(depth_top_center)
    depth_value_mid = np.mean(depth_mid_center)
    depth_value_bottom = np.mean(depth_bottom_center)
    # print(depth_value_top, depth_value_mid, depth_value_bottom)
    return depth_value_top, depth_value_mid, depth_value_bottom

def cal_depth(bounding_boxes, depth_map):
    depth = []
    for i in range(len(bounding_boxes)):
        box = bounding_boxes[i]
        box = box.astype(int)
        # Take the center of the box that covers 50% of the bounding box area
        # height of the pixel 
        box_height = box[3] - box[1]
        # width of the pixel
        box_width = box[2] - box[0]
        padding_value_height = int(0.14*box_height)

        padding_value_width = int(0.14*box_width)

        # new box
        new_box = np.array([box[0]+padding_value_width, box[1]+padding_value_height, box[2]-padding_value_width, box[3]-padding_value_height])

        #calculate the depth
        depth_value = np.mean(depth_map[new_box[1]:new_box[3], new_box[0]:new_box[2]])
        depth.append(depth_value)
    return depth
        
def get_iob(bb1, based_box):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], based_box[0])
    y_top = max(bb1[1], based_box[1])
    x_right = min(bb1[2], based_box[2])
    y_bottom = min(bb1[3], based_box[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    based_box_area = (based_box[2] - based_box[0]) * (based_box[3] - based_box[1])
    object_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])

    if object_area < based_box_area:
        iob = intersection_area / object_area
    else:
        iob = intersection_area / based_box_area

    return iob

# Position boxes

top_left_box = np.array([0,0,213,160])
top_mid_box = np.array([213,0,426,160])
top_right_box = np.array([426,0,640,160])
mid_left_box = np.array([0,160,213,320])
mid_mid_box = np.array([213,160,426,320])
mid_right_box = np.array([426,160,640,320])
bottom_left_box = np.array([0,320,213,480])
bottom_mid_box = np.array([213,320,426,480])
bottom_right_box = np.array([426,320,640,480])

def object_position_find(bounding_box):
    object_position = []
    for i in range(len(bounding_box)):
        box = bounding_box[i]
        box = box.astype(int)
        if get_iob(box, mid_mid_box) > 0.5 or get_iob(box, bottom_mid_box) > 0.5:
            object_position.append('Front')
        elif get_iob(box, mid_left_box) > 0.5:
            object_position.append('Left')
        elif get_iob(box, bottom_left_box) > 0.5:
            object_position.append('Bottom left')
        elif get_iob(box, mid_right_box) > 0.5:
            object_position.append('Right')
        elif get_iob(box, bottom_right_box) > 0.5:
            object_position.append('Bottom Right')
        elif get_iob(box, top_mid_box) > 0.5:
            object_position.append('Top mid')
        elif get_iob(box, top_left_box) > 0.5:
            object_position.append('Top left')
        elif get_iob(box, top_right_box) > 0.5:
            object_position.append('Top right')
        else:
            # Get box center to decide
            box_center_x = (box[0]+box[2])//2
            box_center_y = (box[1]+box[3])//2
            if box_center_x < 213 and box_center_y >= 160 and box_center_y < 320:
                object_position.append('Left')
            elif box_center_x < 213 and box_center_y >= 320:
                object_position.append('Bottom left')
            elif box_center_x >= 213 and box_center_x < 426 and box_center_y >= 160:
                object_position.append('Front')
            elif box_center_x >= 213 and box_center_x < 426 and box_center_y >= 320:
                object_position.append('Bottom')
            elif box_center_x >= 426 and box_center_y >= 160 and box_center_y < 320:
                object_position.append('Right')
            elif box_center_x >= 426 and box_center_y >= 320:
                object_position.append('Bottom right')
            elif box_center_x >= 213 and box_center_x < 426 and box_center_y < 160:
                object_position.append('Above')
            elif box_center_x < 213 and box_center_y < 160:
                object_position.append('Top left')
            elif box_center_x >= 426 and box_center_y < 160:
                object_position.append('Top right')
            else:
                object_position.append('Unknown')
                
    return object_position





#Use for drawing the frame division
depth_center = np.array([320,240])

def object_position_find_2(bounding_box):
    #set depth center as the orgin
    object_position = []
    #using sin cos to find the angle of the object
    for i in range(len(bounding_box)):
        box = bounding_box[i]
        box = box.astype(int)
        box_center_x = (box[0]+box[2])//2
        box_center_y = (box[1]+box[3])//2
        #find the angle of the object
        angle = np.arctan2(box_center_y-depth_center[1],box_center_x-depth_center[0])
        #convert the angle to degree
        angle = np.degrees(angle)
        #print(angle)
        if angle > 45 and angle < 135:
            object_position.append('Top')
        elif angle > 135 or angle < -135:
            object_position.append('Left')
        elif angle > -135 and angle < -45:
            object_position.append('Bottom')
        elif angle > -45 and angle < 45:
            object_position.append('Right')
        else:
            object_position.append('Unknown')
    
    return object_position

def draw_frame_division(img):
    #draw box for the frame division with red color
    img = cv2.rectangle(img,(0,0),(213,160),(0,0,255),2)
    img = cv2.rectangle(img,(213,0),(426,160),(0,0,255),2)
    img = cv2.rectangle(img,(426,0),(640,160),(0,0,255),2)
    img = cv2.rectangle(img,(0,160),(213,320),(0,0,255),2)
    img = cv2.rectangle(img,(213,160),(426,320),(0,0,255),2)
    img = cv2.rectangle(img,(426,160),(640,320),(0,0,255),2)
    img = cv2.rectangle(img,(0,320),(213,480),(0,0,255),2)
    img = cv2.rectangle(img,(213,320),(426,480),(0,0,255),2)
    img = cv2.rectangle(img,(426,320),(640,480),(0,0,255),2)

    #draw text for each box
    img = cv2.putText(img,'Top left',(0,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Top mid',(213,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Top right',(426,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Mid left',(0,175),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Mid mid',(213,175),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Mid right',(426,175),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Bottom left',(0,335),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Bottom mid',(213,335),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Bottom right',(426,335),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    return img

def draw_position_line(img):
    #draw line from top left to bottom right of the square with the center being the frame center
    img = cv2.line(img,(80,0),(560,480),(0,0,255),2)
    #draw line from top right to bottom left of the square with the center being the frame center
    img = cv2.line(img,(560,0),(80,480),(0,0,255),2)
    #write top, bottom, left, right text
    img = cv2.putText(img,'Top',(320,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Bottom',(320,465),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Left',(0,240),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    img = cv2.putText(img,'Right',(600,240),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    return img

def box_center(bounding_box):
    box_center = []
    for i in range(len(bounding_box)):
        box = bounding_box[i]
        # box = box.astype(int)
        box_center_x = (box[0]+box[2])//2
        box_center_y = (box[1]+box[3])//2
        box_center.append([box_center_x,box_center_y])
    return box_center

def draw_position_line_2(img):
    centers = [[50,332]]
    np.array(centers)
    #draw vertical and horizontal line from the center of the frame
    img = cv2.line(img,(320,0),(320,480),(0,0,255),2)
    img = cv2.line(img,(0,240),(640,240),(0,0,255),2)
    #draw line from the center of the frame to the center of the object

    for i in range(len(centers)):

        img = cv2.line(img,(320,240),(centers[i][0], centers[i][1]),(0,0,255),2)
        # write the angle of the object from the center of the frame using euler formula with cos and sin
        #calculate cos to know left or right
        cos = (centers[i][0]-320)/np.sqrt((centers[i][0]-320)**2+(centers[i][1]-240)**2)
        #calculate sin to know top or bottom
        sin = (centers[i][1]-240)/np.sqrt((centers[i][0]-320)**2+(centers[i][1]-240)**2)
        sin =  - sin
        #write the value of cos and sin with 2 decimal places
        img = cv2.putText(img,f'cos: {cos:.2f}',(23,16+75),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
        img = cv2.putText(img,f'sin: {sin:.2f}',(23,16+60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
        #if cos is positive then the object is on the right side of the frame
        if cos > 0:
            img = cv2.putText(img,'Right',(23,16+45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
        #if cos is negative then the object is on the left side of the frame
        else:
            img = cv2.putText(img,'Left',(23,16+45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
        #if sin is positive then the object is on the top side of the frame
        if sin < 0:
            img = cv2.putText(img,'Bottom',(23,16+30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
        #if sin is negative then the object is on the bottom side of the frame
        else:
            img = cv2.putText(img,'Top',(23,16+30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
        
        #Write the length of the line from the center of the frame to the center of the object in pixel
        img = cv2.putText(img,f'Length {np.sqrt((centers[i][0]-320)**2+(centers[i][1]-240)**2):.2f} pixel value',(23,16+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)
    return img

def draw_depth_cal(img):
    #draw two vertical line to show the depth calculation
    img = cv2.line(img,(266,0),(266,480),(0,255,0),2)
    img = cv2.line(img,(373,0),(373,480),(0,255,0),2)

    img = cv2.line(img,(266,0),(373,0),(0,255,0),2)
    img = cv2.line(img,(266,160),(373,160),(0,255,0),2)
    img = cv2.line(img,(266,320),(373,320),(0,255,0),2)
    img = cv2.line(img,(266,480),(373,480),(0,255,0),2)

    return img

     