import cv2
import torch
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # local model
model.classes = [0]
im1 = cv2.imread('Test_img/bucks1.jpg')[..., ::-1] # OpenCV image (BGR to RGB)
res = model([im1])
res.save()