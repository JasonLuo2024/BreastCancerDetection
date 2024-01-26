__author__ = "JasonLuo"
import shutil
import random
import pandas as pd
from tqdm import tqdm
import os
NlHealth_csv = r'D:\Breast_ROI\Metadata\Vindir.csv'

nlhealth_df = pd.read_csv(NlHealth_csv)
import os
from PIL import Image

def process_patient(patient_dir,side_target):
    for file in os.listdir(patient_dir):
        if file.endswith(".png"):
            image = Image.open(os.path.join(patient_dir, file))

            # Flip horizontally (mirror)
            mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)

            mirrored_image.save(os.path.join(side_target, file))

save_path = r'C:\Users\Woody\Desktop\Positive\Vindir'
for subpath, dirs, files in tqdm(os.walk(r'D:\Breast_ROI\Vindir')):
    for file in files:
        file_Path = os.path.join(subpath,file)
        image_Name = file.split('.')[0].split('_')[1]
        filtered_df = nlhealth_df[(nlhealth_df['image_id'] == image_Name)]

        if not filtered_df.empty:
            cancer = filtered_df['breast_birads'].iloc[0]
            laterality = filtered_df['laterality'].iloc[0]
            if (cancer == 'BI-RADS 5' or cancer == 'BI-RADS 4'):
                if (laterality == 'L'):
                    image = Image.open(file_Path)
                    mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    mirrored_image.save(os.path.join(save_path, file))
                else:
                    shutil.copy(file_Path, os.path.join(save_path, file))



