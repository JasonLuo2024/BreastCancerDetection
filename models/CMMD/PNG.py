import os
import pydicom
from tqdm import tqdm
from windowing import windowing
file_path = r'D:\BreastCancer\New folder\manifest-1616439774456\CMMD'
save_folder = r'D:\Breast_ROI\CMMD'

for item in tqdm(os.listdir(file_path)):
    patientId = item
    folder_path = os.path.join(file_path,item)
    for subpath, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                filePath = os.path.join(subpath, file)
                savePath = os.path.join(save_folder, item)
                os.makedirs(savePath, exist_ok=True)
                windowing(filePath, savePath, item)

