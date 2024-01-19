__author__ = "JasonLuo"
import pydicom
import pandas as pd
import os
from tqdm import tqdm
from windowing import windowing
dir = r'D:\BreastCancer\New folder\manifest-1616439774456\CMMD'
csv = r'CMMD_clinicaldata_revision.csv'
df = pd.read_csv(csv)
# filtered_df = df[(df['ID1'] == 'D1-0001') & (df['LeftRight'] == 'R')]
# print(filtered_df)
l_Path = r'C:\Users\Woody\Desktop\All_Positive_Cases\CMMD_L'
r_Path = r'C:\Users\Woody\Desktop\All_Positive_Cases\CMMD_R'

def getCancer(patientID,laterality):
    filtered_df = df[(df['ID1'] == patientID)& (df['LeftRight'] == laterality)]
    if not filtered_df.empty:
        cancer = filtered_df['classification'].iloc[0]
        if (cancer == 'Benign'):
            index = 0
        else:
            index = 1
        return index


for item in tqdm(os.listdir(dir)):
    patientId = item
    folder_path = os.path.join(dir,item)
    for subpath, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                image = os.path.join(subpath, file)
                dcm = pydicom.dcmread(image)
                laterality = dcm[0x0020, 0x0062].value
                cancer = getCancer(patientId, laterality)

                if cancer == 1:
                    save = l_Path if laterality == 'L' else r_Path
                    windowing(image, save,patientId)


