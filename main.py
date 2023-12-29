from model import ObjectDetection, Dist_Depth, depth_onnx_run, object_onnx_run, cal_warning_depth, cal_depth
import cv2

# from utils.utils import draw_detections, final_object_dict
from utils.utils_2 import draw_detections, final_object_dict

# from utils import utils as u1
# from utils import utils_2 as u2

import matplotlib.pyplot as plt
import numpy as np
import time
import pyttsx3
from nicolai import depth_midas

# main_model = Dist_Depth()

object_model = ObjectDetection('model_instances/object_detection/yolov8m_best.onnx')
object_model_2 = ObjectDetection('model_instances/object_detection/yolov8m.onnx')

# start_time = time.time()

if __name__ == '__main__':
    # num_frame = 0
    engine = pyttsx3.init()
    cap = cv2.VideoCapture(0)
    # cap.set(3, 640)
    # cap.set(4, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # num_frame += 1

        # if frame is read correctly ret is True

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break


        # Preprocess the image (use for old depth model)
        
        # Old depth model
        # shape = (480,640,3)
        # frame = cv2.resize(frame, (shape[1] , shape[0]))
        # frame = frame/255
        # depth_map = depth_onnx_run(frame)
        
        # New depth model
        depth_map = depth_midas(frame)

        # Preprocess to display orginal image
        shape = (480,640,3)
        frame = cv2.resize(frame, (shape[1] , shape[0]))
        frame = frame/255
        

        # Object detection
        """
        bounding_box, scores, cls_idx = object_onnx_run(frame, object_model)
        combined_img = draw_detections(frame, bounding_box, scores, cls_idx)
        """
        
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
        depth = cal_warning_depth(depth_map)
        # print(depth)

        iou_threshold = 'SMTH'
        
        #dividing frame in to 9 parts, top left, top mid, top right, mid left, mid mid, mid right, bottom left, bottom mid, bottom right
        #top left
        top_left = frame[0:160, 0:213]
        #top mid
        top_mid = frame[0:160, 213:426]
        #top right
        top_right = frame[0:160, 426:640]
        #mid left
        mid_left = frame[160:320, 0:213]
        #mid mid
        mid_mid = frame[160:320, 213:426]
        #mid right
        mid_right = frame[160:320, 426:640]
        #bottom left
        bottom_left = frame[320:480, 0:213]
        #bottom mid
        bottom_mid = frame[320:480, 213:426]
        #bottom right
        bottom_right = frame[320:480, 426:640]

        # TTS

        # DEPTH_THRESHOLD = 200
        # if depth > DEPTH_THRESHOLD:
        #     # print('Object in front of you!')
        #     engine.say('Object in front of you!')
        #     engine.runAndWait()


        if cv2.waitKey(1) == ord('o'):
            # engine.say('Processing')
            # engine.runAndWait()

            bounding_box_2, scores_2, cls_idx_2 = object_onnx_run(frame, object_model_2)

            # object_dict = u1.final_object_dict(cls_idx)
            object_dict_2 = final_object_dict(cls_idx_2)


            object_dist_2 = cal_depth(bounding_box_2, depth_map)

            combined_img = draw_detections(frame, bounding_box_2, scores_2, cls_idx_2)
            combined_img_depth = draw_detections(depth_map, bounding_box_2, scores_2, cls_idx_2)

            cv2.imshow('Object', combined_img)
            cv2.imshow('Depth_object', combined_img_depth)
            
            for i in range(len(object_dict_2)):
                print(f'{object_dict_2[i]} at: {object_dist_2[i]}')

                # # Distance calculation here?
                # engine.say(object_dict_2[i])
                # engine.runAndWait()

            print('------------------')

            # cv2.imshow('Object', combined_img)


        # When everything done, release the capture
        if cv2.waitKey(1) == ord('q'):

            # print('FPS: ', num_frame/(time.time() - start_time))
            break

    cap.release()
    cv2.destroyAllWindows()