from model import ObjectDetection, Dist_Depth, depth_onnx_run, object_onnx_run
import cv2
from utils.utils import draw_detections, final_object_dict
import matplotlib.pyplot as plt
import numpy as np

# main_model = Dist_Depth()
object_model = ObjectDetection('model_instances/object_detection/yolov8m.onnx')

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # cap.set(3, 640)
    # cap.set(4, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
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
        # cv2.imshow('Object', combined_img)
        """

        if cv2.waitKey(1) == ord('o'):
            bounding_box, scores, cls_idx = object_onnx_run(frame, object_model)
            combined_img = draw_detections(frame, bounding_box, scores, cls_idx)

            object_dict = final_object_dict(cls_idx)
            for i in range(len(object_dict)):
                print(object_dict[i])
            print('------------------')

            # cv2.imshow('Object', combined_img)

        # When everything done, release the capture
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()