from model import ObjectDetection, Dist_Depth, depth_onnx_run, object_onnx_run, cal_warning_depth
import cv2
from utils.utils import draw_detections, final_object_dict
import matplotlib.pyplot as plt
import numpy as np
import time
import pyttsx3
from nicolai import depth_midas

# main_model = Dist_Depth()
object_model = ObjectDetection('model_instances/object_detection/yolov8m.onnx')
# start_time = time.time()
video_path = 'video_test_5.mov'
num_frame = 0

if __name__ == '__main__':
    num_frame += 1
    
    # num_frame = 0
    engine = pyttsx3.init()
    cap = cv2.VideoCapture(video_path)
    # cap.set(3, 640)
    # cap.set(4, 480)

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

        if num_frame % 8 == 0:
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
            # print(type(frame.shape))

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
            
            # cv2.imshow('top left', top_left)

            

            # TTS
            DEPTH_THRESHOLD = 200
            
            # if depth > DEPTH_THRESHOLD:
            #     # print('Object in front of you!')
            #     engine.say('Object in front of you!')
            #     engine.runAndWait()


        if cv2.waitKey(1) == ord('o'):
            if frame.shape != (480,640,3):
                shape = (480,640,3)
                frame = cv2.resize(frame, (shape[1] , shape[0]))
                frame = frame/255

            bounding_box, scores, cls_idx = object_onnx_run(frame, object_model)


            # print(bounding_box)

            combined_img = draw_detections(frame, bounding_box, scores, cls_idx)

            object_dict = final_object_dict(cls_idx)

            cv2.imshow('Object', combined_img)

            

            for i in range(len(object_dict)):
                print(object_dict[i])
                # Distance calculation here?
                engine.say(object_dict[i])
                engine.runAndWait()
                # TTS here?

            print('------------------')

                # cv2.imshow('Object', combined_img)


        # When everything done, release the capture
        if cv2.waitKey(1) == ord('q'):

            # print('FPS: ', num_frame/(time.time() - start_time))
            break

    cap.release()
    cv2.destroyAllWindows()