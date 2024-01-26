__author__ = "JasonLuo"
import os
from tqdm import tqdm
import pandas as pd
import shutil

dir = r'D:\Breast_ROI\NL_Health\Unknown'
csv = r'NL_Health.csv'
df = pd.read_csv(csv)


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


def getViewPosition(image_id,patient_id):
    filtered_df = df[(df['image_id'] == image_id) & (df['patient_id'] == patient_id)]

    if not filtered_df.empty:
        viewPosition = filtered_df['view_position'].iloc[0]

        if filtered_df['laterality'].iloc[0] == 'L' :
            laterality = 'left'
        else:
            laterality = 'right'
        return viewPosition + '_' + laterality
    else:
        return None

def main():

    for subpath, dirs, files in tqdm(os.walk(dir)):
        for file in files:
            file_path = os.path.join(subpath, file)
            patient_id = file.split('_')[3].split('-')[1]
            folder_name = file.split('_IM')[0]

            image_id = file.split('_')[-1].split('.')[0]

            image_type = getViewPosition(image_id,folder_name)

            if image_type != None :
                patient_folder = os.path.join(subpath, patient_id)
                save_path = os.path.join(patient_folder, image_type)
                os.makedirs(save_path, exist_ok=True)
                shutil.move(file_path, save_path)




main()