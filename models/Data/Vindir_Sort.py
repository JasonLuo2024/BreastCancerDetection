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
            if (cancer == 'BI-RADS 5' or cancer == 'BI-RADS 4'):
                index = 1
            else:
                index = 0

            return index,split,viewPosition


    except:
        return False, False



for subpath, dirs, files in os.walk(r'D:\thesis\Vindir_ROI'):
    for file in files:
        image_path = os.path.join(subpath,file)
        index,split,viewPosition = getCancer(file)
        if split == 'training':
            new_Image_Path = os.path.join(r'D:\thesis\train',file)
            shutil.copy(image_path,new_Image_Path)
        else:
            new_Image_Path = os.path.join(r'D:\thesis\test', file)
            shutil.copy(image_path, new_Image_Path)

