from model import ObjectDetection, Dist_Depth, depth_onnx_run, object_onnx_run
import cv2
from utils.utils_2 import draw_detections, final_object_dict, cal_depth, object_position_find
import matplotlib.pyplot as plt
import numpy as np
import time
import pyttsx3
from nicolai import depth_midas

object_model = ObjectDetection('model_instances/object_detection/yolov8m_weight.onnx')
# start_time = time.time()
video_path = 'videos/video_1.mp4'
num_frame = 0

if __name__ == '__main__':
    num_frame += 1
    engine = pyttsx3.init()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        num_frame += 1

        # if frame is read correctly ret is True

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if num_frame % 7 == 0:  
            depth_map, original_depth_map = depth_midas(frame)

            # Preprocess to display orginal image
            shape = (480,640,3)
            frame = cv2.resize(frame, (shape[1] , shape[0]))
            frame = frame/255
            
            # Display the resulting frame

            cv2.imshow('Depth', depth_map)
            cv2.imshow('Normal', frame)

            """
            cv2.imshow('Object', combined_img)
            """

            # If too close then display warning
            # TTS
            """
            TESTING
            """


        if cv2.waitKey(1) == ord('o'):
            if frame.shape != (480,640,3):
                shape = (480,640,3)
                frame = cv2.resize(frame, (shape[1] , shape[0]))
                frame = frame/255

            bounding_box, scores, cls_idx = object_onnx_run(frame, object_model)
            object_position = object_position_find(bounding_box)
            object_dict_2 = final_object_dict(cls_idx)

            object_dist_2, object_dist_median, object_dist_max, object_dist_min = cal_depth(bounding_box, original_depth_map)

            combined_img = draw_detections(frame, bounding_box, scores, cls_idx)
            combined_img_depth = draw_detections(depth_map, bounding_box, scores, cls_idx)

            cv2.imshow('Depth_Box', combined_img_depth)
            cv2.imshow('Object', combined_img)

            for i in range(len(object_dict_2)):
                print(f'{object_dict_2[i]} at: mean {object_dist_2[i]}, median {object_dist_median[i]}, max {object_dist_max[i]}, min {object_dist_min[i]}: {object_position[i]}')
                # TTS here?
            print('------------------')
            cv2.waitKey(0)

        # When everything done, release the capture
        if cv2.waitKey(1) == ord('q'):

            # print('FPS: ', num_frame/(time.time() - start_time))
            break

    cap.release()
    cv2.destroyAllWindows()