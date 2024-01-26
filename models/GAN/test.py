__author__ = "JasonLuo"
import os
from PIL import Image

# Base directory where the patient folders are located
def process_patient(patient_dir,side_target):
    for file in os.listdir(patient_dir):
        if file.endswith(".png"):
            image = Image.open(os.path.join(patient_dir, file))

            # Flip horizontally (mirror)
            mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)

            mirrored_image.save(os.path.join(side_target, file))

datasetPath = r'C:\Users\Woody\Desktop\Positive\Vindir'
for item in os.listdir(datasetPath):
    folderPath = os.path.join(datasetPath, item)
    for subfolder in ['CC_right', 'MLO_right']:
        subPath = os.path.join(folderPath, subfolder)
        for filename in os.listdir(subPath):
            filePath = os.path.join(subPath,filename)
            image = Image.open(filePath)
            mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mirrored_image.save(filePath)
