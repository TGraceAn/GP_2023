from model import ObjectDetection, object_onnx_run
from utils.utils_2 import draw_detections
import cv2


object_model_2 = ObjectDetection('model_instances/object_detection/yolov8m_weight.onnx')

image_path = 'images/image_27.jpg'
image = cv2.imread(image_path)
shape = (480,640,3)
image = cv2.resize(image, (shape[1] , shape[0]))
image = image/255

bounding_box_2, scores_2, cls_idx_2 = object_onnx_run(image, object_model_2)
combined_img = draw_detections(image, bounding_box_2, scores_2, cls_idx_2)

cv2.imshow('Object', combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()