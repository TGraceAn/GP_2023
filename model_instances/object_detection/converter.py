import torch
model = torch.load('model_instances/object_detection/best.pt')
model.eval()

