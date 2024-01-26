__author__ = "JasonLuo"
import torch
import glob
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
model = torch.hub.load(r'D:\breastcancerdetection\image_processing\yolov5', 'custom', path=r'D:\breastcancerdetection\image_processing\ROI_models\rsna-roi-003.pt', source='local')

directory = r'D:\Breast_ROI\NL_Health'
traget = r'D:\Breast_ROI\NL_Health_ROI'

image_list = []
for subpath, dirs, files in tqdm(os.walk(directory)):
    for file in files:
        image_path = os.path.join(subpath,file)
        image_list.append(image_path)
print(len(image_list))

images = []
for img_file in image_list:


    image_name = os.path.basename(img_file).split('.')[0]
    # Read file from file
    frame = cv2.imread(img_file)

    # Make prediction
    detections = model(frame)
    for idx, box in enumerate(detections.xyxy[0]):
        x1, y1, x2, y2 = map(int, box[:4])

        # Crop the ROI from the frame
        roi = frame[y1:y2, x1:x2]

        file_name = os.path.join(traget, f"{image_name}.png")
        cv2.imwrite(file_name, roi)


