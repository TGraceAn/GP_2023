'''
- train model with difference dataset --> robust
- open webcam with openCV, combine with pytorch, load in the models and do depth estimation 
- on live webcam feed 
'''

import cv2
import torch
import time
import numpy as np



# Load a MiDas model for depth estimation
#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# hybrid: mobile devices
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# small: low_budget computer --> the media smart model 

model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
# load img --> numeric --> resize, normalize the images --> same input format 
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    # based on transformers, have different kind of input format 
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
    
depth_shape = (480,640)

def depth_midas(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # print(input_batch.shape) --> torch.Size([1, 3, 128, 256])

    with torch.no_grad():
        prediction = midas(input_batch)
        
        # model's output --> not normalized, return original shape ... --> openCV image  
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=depth_shape,
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()
        orginal_depth_map = depth_map.copy()

        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        # normalization
        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
        
        return depth_map, orginal_depth_map