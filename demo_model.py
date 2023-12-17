import onnxruntime as nxrun
from PIL import Image
# from opyv8 import Predictor

model = nxrun.InferenceSession('model_instances/object_detection/yolov8m.onnx', providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
# List of classes where the index match the class id in the ONNX network
classes =  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def object_onnx_run(img):
    
    arr = np.expand_dims(img, 0)
    arr = np.array(arr, dtype = np.float32)
    arr = np.array(np.transpose(arr, (0, 3, 1, 2)), dtype=np.float32)

    model_inputs = Object_SESS.get_inputs()

    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_shape = model_inputs[0].shape
    input_height = input_shape[2]
    input_width = input_shape[3]

    
    # bounding_box, scores, cls = model(arr)
    
    return bounding_box, scores, cls