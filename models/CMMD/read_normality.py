__author__ = "JasonLuo"
import pandas as pd
import os

csv =  r'D:\Breast_ROI\Metadata\Vindir.csv'
df = pd.read_csv(csv)
dir = r'D:\Vindir\images'


def getCancer(patient_ID):
    filtered_df = df[df['image_id'] == patient_ID]
    if not filtered_df.empty:
        cancer = filtered_df['breast_birads'].iloc[0]
        if (cancer == 'BI-RADS 5' or cancer == 'BI-RADS 4'):
            index = 1
        else:
            index = 0
        return index



patient_case = []
for item in os.listdir(dir):
    patient_folder = os.path.join(dir,item)
    for file in os.listdir(patient_folder):
        if file.endswith(".dicom"):
            patient_ID = file.split('.')[0]
            cancer = getCancer(patient_ID)
            print(cancer)





