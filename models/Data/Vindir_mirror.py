__author__ = "JasonLuo"
__author__ = "JasonLuo"
import os
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import shutil

rsna_csv = r'D:\Breast_ROI\Metadata\RSNA.csv'
vindir_csv = r'D:\Breast_ROI\Metadata\Vindir.csv'
NlHealth_csv = r'D:\Breast_ROI\Metadata\NL_Health.csv'

rsna_df = pd.read_csv(rsna_csv)
vindir_df = pd.read_csv(vindir_csv)
nlhealth_df = pd.read_csv(NlHealth_csv)

def getCancer(image):
    try:
        image_id = image.split('_')[1].split('.')[0]
        filtered_df = vindir_df[vindir_df['image_id'] == image_id]
        if not filtered_df.empty:
            cancer = filtered_df['breast_birads'].iloc[0]
            split = filtered_df['split'].iloc[0]
            viewPosition = filtered_df['view_position'].iloc[0]
            laterality = filtered_df['laterality'].iloc[0]
            if (cancer == 'BI-RADS 5' or cancer == 'BI-RADS 4'):
                index = 1
            else:
                index = 0

            return index,split,viewPosition,laterality


    except:
        return False, False



for subpath, dirs, files in os.walk(r'C:\Users\Woody\Desktop\New folder (4)\cbf65b79b18fcec0f20a696e65261936_57e1e5a007ddaa1b61acb3f975ff8fef.png'):
    for file in files:
        try:
            image_path = os.path.join(subpath, file)
            index, split, viewPosition, laterality = getCancer(file)
            if laterality == 'R':
                image = Image.open(image_path)

                # Flip horizontally (mirror)
                mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)

                mirrored_image.save(image_path)
        except Exception as e:
            print(f'{e}')



