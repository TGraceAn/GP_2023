import cv2 as cv
from model import ObjectDetection, object_onnx_run
from utils.utils_2 import draw_detections, final_object_dict, object_position_find

object_model = ObjectDetection('model_instances/object_detection/yolov8n_weight.onnx')
if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv.flip(frame,1)
        frame = cv.resize(frame, (640,480))

        boxes, scores, cls_ids = object_onnx_run(frame, object_model)
        print(final_object_dict(cls_ids))
        print(object_position_find(boxes))

        print(boxes)
        print(scores)
        print(cls_ids)
        break
        frame = draw_detections(frame, boxes, scores, cls_ids)

        cv.imshow('BB', frame)

        if cv.waitKey(1) == ord('q'): break

    cap.release()
    cv.destroyAllWindows()