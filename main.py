from model import ObjectDetection, Dist_Depth, depth_onnx_run, object_onnx_run, cal_depth
import cv2
from utils.utils import draw_detections, final_object_dict
import matplotlib.pyplot as plt
import numpy as np
import time
import pyttsx3

# main_model = Dist_Depth()
object_model = ObjectDetection('model_instances/object_detection/yolov8m.onnx')
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


        # Preprocess the image
        shape = (480,640,3)
        frame = cv2.resize(frame, (shape[1] , shape[0]))
        frame = frame/255

        depth_map = depth_onnx_run(frame)

        # Object detection
        """
        bounding_box, scores, cls_idx = object_onnx_run(frame, object_model)
        combined_img = draw_detections(frame, bounding_box, scores, cls_idx)
        """
        # Display the resulting frame

        cv2.imshow('Depth', depth_map)
        """
        cv2.imshow('Object', combined_img)
        """

        # If too close then display warning
        # TTS
        """
        TESTING
        """
        depth = cal_depth(depth_map)
        print(depth)

        THRESHOLD = 100
        if depth < THRESHOLD:
            # print('Object in front of you!')
            engine.say('Object in front of you!')
            engine.runAndWait()
            # TTS here?

        if cv2.waitKey(1) == ord('o'):
            bounding_box, scores, cls_idx = object_onnx_run(frame, object_model)
            combined_img = draw_detections(frame, bounding_box, scores, cls_idx)

            object_dict = final_object_dict(cls_idx)
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
