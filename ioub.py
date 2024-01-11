import cv2 as cv
import numpy as np
from nicolai import depth_midas
from model import ObjectDetection, object_onnx_run
from utils.utils_2 import *

INPUT_SHAPE = 640, 480 # width, height
FILL_COLOR = 255, 255, 255
DEPTH_THRESHOLD = 700
OBJECT_THRESHOLD = 550
K = 5 # for EMA
object_model = ObjectDetection('model_instances/object_detection/yolov8s_weight.onnx')

def get_descriptions(obj_dict, obj_dist, obj_position):
    obj_descriptions = {}
    for obj, dist, pos in zip(obj_dict, obj_dist, obj_position):
        descriptions = f"at: mean {dist}, {pos}, "
        descriptions += 'CLOSE' if dist > OBJECT_THRESHOLD else 'FAR'

        obj_descriptions[obj] = descriptions

    return obj_descriptions

if __name__ == '__main__':
    video_source = 0 # can be mp4 file
    cap = cv.VideoCapture(video_source)

    i, ema = 0, None
    while cap.isOpened():
        ret, frame = cap.read()
        if video_source == 0:
            frame = cv.flip(frame, 1)
        frame = cv.resize(frame, INPUT_SHAPE)
        key = cv.waitKey(1)

        # CHEAT
        x1, y1, x2, y2 = 0, 320, 160, 480
        cheat_frame = frame.copy()
        cheat_frame[y1:y2, x1:x2] = FILL_COLOR

        depth = depth_midas(cheat_frame)
        # DISPLAY FRAME AND DEPTH
        display = cv.normalize(
            depth, None, 0, 255,
            norm_type=cv.NORM_MINMAX,
            dtype=cv.CV_8U
        )
        display = cv.applyColorMap(display, cv.COLORMAP_MAGMA)

        cv.imshow('depth', display)
        cv.imshow('frame', cheat_frame)

        cv.moveWindow('depth', 0, 0)
        cv.moveWindow('frame', 700, 0)

        if key == ord('q'): break

        # WARNING
        warning_depth = cal_warning_depth(depth)
        if i == 0: ema = np.zeros_like(warning_depth)
        if i < K:
            i += 1
            ema += warning_depth
            if i == K: ema /= K
            continue

        C = 2/(K+1)
        ema = warning_depth * C + ema * (1-C)
        # print(ema)
        if (ema > DEPTH_THRESHOLD).any():
            print('Object in front of you!')

        # OBJECT DESCRIPTION
        if key == ord('o'):
            cv.destroyWindow('detect')
            boxes, scores, cls_ids = object_onnx_run(frame, object_model)

            object_dict = final_object_dict(cls_ids)
            object_position = object_position_find(boxes)
            object_dist = cal_depth(boxes, depth)

            obj_descriptions = get_descriptions(
                object_dict,
                object_dist,
                object_position
            )
            print(obj_descriptions)

            detect = draw_detections(display, boxes, scores, cls_ids)
            cv.imshow('detect', detect)
            cv.moveWindow('detect', 350, 100)

    cap.release()
    cv.destroyAllWindows()