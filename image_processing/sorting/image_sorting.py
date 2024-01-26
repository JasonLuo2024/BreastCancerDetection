__author__ = "JasonLuo"
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import shutil

def getImagePath(patientID, Laterality, ViewPosition):
    filtered_df = df[
    (df['patient_id'] == int(patientID)) &
    (df['laterality'] == Laterality) &
    (df['view'] == ViewPosition)
]
    filtered_df = filtered_df.drop_duplicates(subset=['image_id', 'laterality', 'view'], keep='first')

    if not filtered_df.empty:
        cancer = filtered_df['cancer'].iloc[0]
        image_ID = filtered_df['image_id'].iloc[0]
        image_path = str(patientID) + '_' + str(image_ID) + '.png'

        return image_path,cancer

train_csv = r'C:\Users\Woody\Desktop\metaData\train.csv'
df = pd.read_csv(train_csv)
directory = r'D:\Breast_ROI\RSNA'
for root, directories, files in os.walk(directory):
    for dir in directories:
        Patient_id = dir
        folderPath = os.path.join(root, dir)
        CC_left_image, CC_left_label = getImagePath(Patient_id, 'L', 'CC')

        Left_CC_path = os.path.join(folderPath,'CC_left')
        os.makedirs(Left_CC_path, exist_ok=True)
        shutil.move(os.path.join(folderPath, CC_left_image), Left_CC_path)

        CC_right_image, CC_right_label = getImagePath(Patient_id, 'R', 'CC')

        Right_CC_path = os.path.join(folderPath, 'CC_right')
        os.makedirs(Right_CC_path , exist_ok=True)
        shutil.move(os.path.join(folderPath, CC_right_image), Right_CC_path)


        MLO_left_image, MLO_left_label = getImagePath(Patient_id, 'L', 'MLO')

        Left_MLO_path = os.path.join(folderPath, 'MLO_left')
        os.makedirs(Left_MLO_path, exist_ok=True)
        shutil.move(os.path.join(folderPath, MLO_left_image), Left_MLO_path)


        MLO_right_image, MLO_right_label = getImagePath(Patient_id, 'R', 'MLO')

        Right_MLO_path = os.path.join(folderPath, 'MLO_right')
        os.makedirs(Right_MLO_path, exist_ok=True)
        shutil.move(os.path.join(folderPath, MLO_right_image), Right_MLO_path)
