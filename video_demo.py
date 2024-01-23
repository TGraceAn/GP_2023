import cv2 as cv
import numpy as np
from nicolai import depth_midas
from model import ObjectDetection, object_onnx_run
from utils.utils_2 import *
from run_onnx import run
import time

model_path = 'model_instances/dist_depth/model-small.onnx'

INPUT_SHAPE = 640, 480 # width, height
FILL_COLOR = 255, 255, 255
DEPTH_THRESHOLD = 650
OBJECT_THRESHOLD = 400
K = 5 # for EMA

# count = 0
# start_time = time.time()

object_model = ObjectDetection('model_instances/object_detection/yolov8s_weight.onnx')

obj = cv2.VideoWriter('output_videos/obj_3.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, INPUT_SHAPE)

depth_obj = cv2.VideoWriter('output_videos/depth_obj_3.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, INPUT_SHAPE)

obj_warning = cv2.VideoWriter('output_videos/obj_warning_3.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, INPUT_SHAPE)

obj_des = cv2.VideoWriter('output_videos/obj_descriptions_3.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, INPUT_SHAPE)


def get_displayed_depth(depth):
    display = cv.normalize(
        depth, None, 0, 255,
        norm_type=cv.NORM_MINMAX,
        dtype=cv.CV_8U
    )
    display = cv.applyColorMap(display, cv.COLORMAP_MAGMA)
    return display

if __name__ == '__main__':
    video_source = 'videos/lobby.mp4'
    # can be mp4 file
    cap = cv.VideoCapture(video_source)

    if not cap.isOpened():
        print("End video camera")
        cap.release()
        obj.release()
        depth_obj.release()
        obj_warning.release()
        obj_des.release()

        cv.destroyAllWindows()
        exit()
    else:
        i, ema = 0, None
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv.resize(frame, INPUT_SHAPE)
            key = cv.waitKey(1)

            # CHEAT
            x1, y1, x2, y2 = 0, 320, 160, 480
            cheat_frame = frame.copy()
            cheat_frame[y1:y2, x1:x2] = FILL_COLOR

            depth = depth_midas(cheat_frame)

            # DISPLAY FRAME AND DEPTH
            display = get_displayed_depth(depth)

            if key == ord('q'): 
                break

            # WARNING
            warning_depth, additional_depth, additional_depth_2 = cal_warning_depth(depth)
                
            if (warning_depth > DEPTH_THRESHOLD).any() or additional_depth_2 > DEPTH_THRESHOLD:
                print('Object in front of you!')
                display = cv.putText(display, 'Object in front of you!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            display = cv.putText(display, f'Top mean: {warning_depth[0]:.2f}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            display = cv.putText(display, f'Mid mean: {warning_depth[1]:.2f}', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            display = cv.putText(display, f'Bottom mean: {warning_depth[2]:.2f}', (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            display = cv.putText(display, f'Half bottom mean: {additional_depth_2:.2f}', (50, 250), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            
            obj_warning.write(display)
            cv.imshow('depth_warning', display)


            depth = depth_midas(frame)
            boxes, scores, cls_ids = object_onnx_run(frame, object_model)
            obj_dict = final_object_dict(cls_ids)
            obj_position = object_position_find(boxes)
            obj_dist = cal_depth(boxes, depth)

            dist_bool = ['CLOSE' if d > OBJECT_THRESHOLD else 'FAR'
                for d in obj_dist]

            obj_descriptions = list(zip(obj_dict, obj_position, dist_bool))

            from collections import Counter
            depth_obj_des = get_displayed_depth(depth)
            display = depth_obj_des.copy()

            final_description = []
            for description, n in Counter(obj_descriptions).items():
                print(f"{n} {' '.join(description)}")
                final_description.append(f"{n} {' '.join(description)}")

            for i in range(len(final_description)):
                depth_obj_des = cv.putText(depth_obj_des, final_description[i], (25, 25*(i+1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)

            detect = draw_detections(display, boxes, scores, cls_ids)

            frame_2 = frame.copy()
            frame = draw_detections(frame, boxes, scores, cls_ids)

            obj.write(frame)
            depth_obj.write(detect)
            obj_des.write(depth_obj_des)

            cv.imshow('depth_obj', depth_obj_des)
            cv.imshow('detect', detect)
            cv.imshow('object', frame)         

            # OBJECT DESCRIPTION
            if key == ord('o'):
                pass
