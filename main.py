from model import ObjectDetection, object_onnx_run_2
import cv2
from utils.utils_2 import draw_detections, final_object_dict, cal_warning_depth, cal_depth, object_position_find
import time
import pyttsx3
from nicolai_2 import depth_midas

# import matplotlib.pyplot as plt
# import numpy as np

object_model_2 = ObjectDetection('model_instances/object_detection/yolov8s_weight.onnx')
# start_time = time.time()
# Front camera is flipped
# count = 1

if __name__ == '__main__':
    # num_frame = 0
    engine = pyttsx3.init()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        # num_frame += 1
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        shape = (480,640,3)
        frame = cv2.resize(frame, (shape[1] , shape[0]))

        frame_modified = frame.copy()
        x1, y1, x2, y2 = 0, 320, 160, 480
        frame_modified[y1:y2, x1:x2] = [255, 255, 255]

        # New depth model
        depth_map, original_depth_map = depth_midas(frame_modified)

        # Display the resulting frame

        cv2.imshow('Normal_modified', frame_modified)
        cv2.imshow('Depth', depth_map)
        cv2.imshow('Normal', frame)

        # If too close then display warning
        # TTS
        depth_value_top, depth_value_mid, depth_value_bottom = cal_warning_depth(original_depth_map)
        

        # TTS
        DEPTH_THRESHOLD = 650
        if depth_value_top > DEPTH_THRESHOLD or depth_value_mid > DEPTH_THRESHOLD or depth_value_bottom > DEPTH_THRESHOLD:
            print('Object in front of you!')
            print("Top: ", depth_value_top, "Mid: ", depth_value_mid, "Bottom: ", depth_value_bottom)
        #     engine.say('Object in front of you!')
        #     engine.runAndWait()

        OBJECT_THRESHOLD = 400

        # Use for describing scene
        if cv2.waitKey(1) == ord('o'):
            depth_map, original_depth_map = depth_midas(frame)
            frame = frame/255
            bounding_box_2, scores_2, cls_idx_2 = object_onnx_run_2(frame, object_model_2)
            object_position = object_position_find(bounding_box_2)
            object_dict_2 = final_object_dict(cls_idx_2)
            # Testing position of object
            # If object's bounding box fits 50% of the center frame then consider it as the object in front of the user

            object_dist_2 = cal_depth(bounding_box_2, original_depth_map)

            combined_img = draw_detections(frame, bounding_box_2, scores_2, cls_idx_2)
            combined_img_depth = draw_detections(depth_map, bounding_box_2, scores_2, cls_idx_2)
            
            cv2.imshow('Object', combined_img)
            cv2.imshow('Depth_object', combined_img_depth)

            # #Use for report
            # frame = cv2.convertScaleAbs(frame, alpha=(255.0))
            # combined_img = cv2.convertScaleAbs(combined_img, alpha=(255.0))
            # cv2.imwrite('images/normal.jpg', frame)
            # cv2.imwrite('images/depth.jpg', depth_map)
            # cv2.imwrite('images/object.jpg', combined_img)
            # cv2.imwrite('images/depth_object.jpg', combined_img_depth)
            # print(bounding_box_2[0], object_dict_2[0], object_dist_2[0])
            
            for i in range(len(object_dict_2)):
                print(f'{object_dict_2[i]} at: mean {object_dist_2[i]}  {object_position[i]}')
                ('------------------')
            
            for i in range(len(object_dict_2)):
                if object_dist_2[i] > OBJECT_THRESHOLD:
                    print(f"{object_dict_2[i]} is close to the {object_position[i]}")
                    # engine.say(f"{object_dict_2[i]} is close to {object_position[i]} of you")
                    # engine.runAndWait()
                else :
                    print(f"{object_dict_2[i]} is far to the {object_position[i]}")
                    # engine.say(f"{object_dict_2[i]} is far from {object_position[i]} of you")
                    # engine.runAndWait()
            print('------------------')

        if cv2.waitKey(1) == ord('q'):
            # print('FPS: ', num_frame/(time.time() - start_time))

            break

    cap.release()
    cv2.destroyAllWindows()