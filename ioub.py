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

count = 1
def get_displayed_depth(depth):
    display = cv.normalize(
        depth, None, 0, 255,
        norm_type=cv.NORM_MINMAX,
        dtype=cv.CV_8U
    )
    display = cv.applyColorMap(display, cv.COLORMAP_MAGMA)
    return display

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
        display = get_displayed_depth(depth)

        cv.imshow('depth', display)
        cv.imshow('frame', cheat_frame)

        cv.moveWindow('depth', 0, 0)
        cv.moveWindow('frame', 700, 0)

        if key == ord('q'): break

        # WARNING
        warning_depth, additional_depth = cal_warning_depth(depth)

        if i == 0: ema = np.zeros_like(warning_depth)
        if i < K:
            i += 1
            ema += warning_depth
            if i == K: ema /= K
        else:
            C = 2/(K+1)

            ema = warning_depth * C + ema * (1-C)
            print(warning_depth)

            if (warning_depth > DEPTH_THRESHOLD).any() or additional_depth > DEPTH_THRESHOLD:
                print('Object in front of you!')
                
                if count % 5 == 0:
                    # write on display with only 2 decimal places
                    display = cv.putText(display, 'Object in front of you!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                    display = cv.putText(display, f'Top mean: {warning_depth[0]:.2f}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                    display = cv.putText(display, f'Mid mean: {warning_depth[1]:.2f}', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                    display = cv.putText(display, f'Bottom mean: {warning_depth[2]:.2f}', (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                    display = cv.putText(display, f'Flat bottom mean: {additional_depth:.2f}', (50, 250), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                    cv.imwrite(f'output_images/normal_{count}.jpg', frame)
                    cv.imwrite(f'output_images/warning_{count}.jpg', display)
                count += 1
                

        # OBJECT DESCRIPTION
        if key == ord('o'):
            depth = depth_midas(frame)
            boxes, scores, cls_ids = object_onnx_run(frame, object_model)

            obj_dict = final_object_dict(cls_ids)
            obj_position = object_position_find(boxes)
            obj_dist = cal_depth(boxes, depth)

            dist_bool = ['CLOSE' if d > OBJECT_THRESHOLD else 'FAR'
                for d in obj_dist]

            obj_descriptions = list(zip(obj_dict, obj_position, dist_bool))
            from collections import Counter
            for description, n in Counter(obj_descriptions).items():
                print(f"{n} {' '.join(description)}")

            display = get_displayed_depth(depth)
            detect = draw_detections(display, boxes, scores, cls_ids)

            cv.imshow('detect', detect)
            cv.moveWindow('detect', 600, 200)

    cap.release()
    cv.destroyAllWindows()