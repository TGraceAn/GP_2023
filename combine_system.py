import cv2 as cv
import numpy as np
from nicolai import depth_midas
from model import ObjectDetection, object_onnx_run
from utils.utils_2 import *
from run_onnx import run

model_path = 'model_instances/dist_depth/model-small.onnx'

INPUT_SHAPE = 640, 480 # width, height
FILL_COLOR = 255, 255, 255
DEPTH_THRESHOLD = 650
OBJECT_THRESHOLD = 400
K = 5 # for EMA
object_model = ObjectDetection('model_instances/object_detection/yolov8s_weight.onnx')

def get_displayed_depth(depth):
    display = cv.normalize(
        depth, None, 0, 255,
        norm_type=cv.NORM_MINMAX,
        dtype=cv.CV_8U
    )
    display = cv.applyColorMap(display, cv.COLORMAP_MAGMA)
    return display

image_path = 'images4report/depth.jpg'
image = cv.imread(image_path)

if __name__ == '__main__':
    image = cv.resize(image, INPUT_SHAPE)
    # CHEAT
    x1, y1, x2, y2 = 0, 320, 160, 480
    cheat_frame = image.copy()
    cheat_frame[y1:y2, x1:x2] = FILL_COLOR

    depth = depth_midas(cheat_frame)
    # DISPLAY FRAME AND DEPTH
    display = get_displayed_depth(depth)

    cv.imshow('depth', display)
    cv.imshow('frame', cheat_frame)

    cv.moveWindow('depth', 0, 0)
    cv.moveWindow('frame', 700, 0)

    # # WARNING
    # depth_value_top, depth_value_mid, depth_value_bottom = cal_warning_depth(depth)
    # if  depth_value_top > DEPTH_THRESHOLD or depth_value_mid > DEPTH_THRESHOLD or depth_value_bottom > DEPTH_THRESHOLD:
    #     print('Object in front of you!')
        

    # OBJECT DESCRIPTION
    depth = depth_midas(image)
    boxes, scores, cls_ids = object_onnx_run(image, object_model)

    obj_dict = final_object_dict(cls_ids)
    obj_position = object_position_find(boxes)
    obj_dist = cal_depth(boxes, depth)

    dist_bool = ['CLOSE' if d > OBJECT_THRESHOLD else 'FAR'
        for d in obj_dist]

    obj_descriptions = list(zip(obj_dict, obj_position, dist_bool))
    from collections import Counter
    for description, n in Counter(obj_descriptions).items():
        print(f"{n} {' '.join(description)}")

    # DISPLAY
    detect_orginal = draw_detections(image, boxes, scores, cls_ids)
    display = get_displayed_depth(depth)
    detect = draw_detections(display, boxes, scores, cls_ids)
    
    cv.imwrite('output_images/detect.jpg', detect)
    # cv.imwrite('output_images/depth.jpg', display)
    cv.imwrite('output_images/detect_orginal.jpg', detect_orginal)


    cv.imshow('detect', detect)
    cv.moveWindow('detect', 600, 200)
    cv.waitKey(0)
    cv.destroyAllWindows()
