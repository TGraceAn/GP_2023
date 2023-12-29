import time
import numpy as np
import torch
from torch import nn
import numpy as np
import onnxruntime as nxrun
import os
import sys
import cv2
import numpy as np
from utils.utils import multiclass_nms, xywh2xyxy

"""
Object dection model

"""
Object_SESS = nxrun.InferenceSession('model_instances/object_detection/yolov8m.onnx', providers=['AzureExecutionProvider', 'CPUExecutionProvider'])


class ObjectDetection:
    def __init__(self, path, conf_thres=0.25, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = nxrun.InferenceSession(path,
                providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, input_tensor):
        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

############################################################################################################
#TODO: Finish the rest of the models
# Testing new code
"""
def object_onnx_run_2(img):
    
    arr = np.expand_dims(img, 0)
    arr = np.array(arr, dtype = np.float32)
    arr = np.array(np.transpose(arr, (0, 3, 1, 2)), dtype=np.float32)

    model_inputs = Object_SESS.get_inputs()

    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_shape = model_inputs[0].shape
    input_height = input_shape[2]
    input_width = input_shape[3]

    outputs = Object_SESS.run(input_names, {input_names[0]: arr})
    # bounding_box, scores, cls = model(arr)
    
    return bounding_box, scores, cls
"""
    
def object_onnx_run(img, model):
    
    arr = np.expand_dims(img, 0)
    arr = np.array(arr, dtype = np.float32)
    arr = np.array(np.transpose(arr, (0, 3, 1, 2)), dtype=np.float32)

    bounding_box, scores, cls = model(arr)
    
    return bounding_box, scores, cls


"""
Dist_Depth model in seperation for the flow

"""
Dist_SESS = nxrun.InferenceSession('model_instances/dist_depth/dist_depth.onnx', providers=['AzureExecutionProvider', 'CPUExecutionProvider'])

def depth_onnx_run(img):

    img = np.array(img, dtype = np.float32)
    arr = np.expand_dims(img, 0)

    input_name = Dist_SESS.get_inputs()[0].name

    # label_name = sess.get_outputs()[0].name
    
    out = Dist_SESS.run(None, {input_name: arr})[0]
    out = out[0, :, :, 0]
    out = (out/out.max()*255).astype(np.uint8)

    return out




class Dist_Depth(nn.Module):
    def __init__(self, shape = (480,640,3)):
        super(Dist_Depth, self).__init__()
        self.shape = shape

    def __onnx_run__(self, inp: np.ndarray, weight_path = 'model_instances/dist_depth/dist_depth.onnx'):
        sess = nxrun.InferenceSession(weight_path, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: inp})[0]
        return out

    def __prep__(self, inp):
        arr = np.expand_dims(inp, 0)
        arr = np.array(arr, dtype = np.float32)
        # arr = cv2.resize(arr, (self.shape[1] , self.shape[0]))
        # arr = arr/255
        return arr
        
    def forward(self, x):
        inp = self.__prep__(x)
        # Two models need two different shapes

        depth_map = self.__onnx_run__(inp)
        depth_map = depth_map[0,:,:,0]
        depth_map = (depth_map/depth_map.max()*255).astype(np.uint8)

        return depth_map





"""
Integration class for the two models
Explanation here:

"""
class Integration(nn.Module):
    def __init__(self):
        super(Integration, self).__init__()
        self.obj_detect = ObjectDetection('model_instances/object_detection/yolov8m.onnx')

    def __onnx_run__(self, inp: np.ndarray, weight_path = 'model_instances/dist_depth/dist_depth.onnx'):
        sess = nxrun.InferenceSession(weight_path, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: inp})[0]
        return out
    
    def __prep__(self, inp):
        arr = np.expand_dims(inp, 0)
        arr = np.array(arr, dtype = np.float32)


        return arr
        
    def forward(self, x):
        inp = self.__prep__(x)
        # print(inp.shape)
        
        # Two models need two different shapes
        arr = np.array(np.transpose(inp, (0, 3, 1, 2)), dtype=np.float32)
        # print(arr.shape)

        bounding_box, scores, cls = self.obj_detect(arr)
        depth_map = self.__onnx_run__(inp)

        depth_map = depth_map[0,:,:,0]
        depth_map = (depth_map/depth_map.max()*255).astype(np.uint8)

        return bounding_box, scores, cls, depth_map
    
def cal_warning_depth(depth_map):
    # Take the mean of the 63x47 center of the depth map 640x480
    depth = depth_map[216:263, 288:335]
    depth = np.mean(depth)
    return depth

def cal_depth(bounding_boxes, depth_map):
    depth = []
    for i in range(len(bounding_boxes)):
        box = bounding_boxes[i]
        box = box.astype(int)
        depth_value = np.mean(depth_map[box[1]:box[3], box[0]:box[2]])
        depth.append(depth_value)
    return depth
