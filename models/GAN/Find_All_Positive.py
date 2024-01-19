import shutil
import os
import pandas as pd
# # Define the source and destination paths
# source_path = 'path/to/your/source/image.jpg'
# destination_path = 'path/to/your/destination/folder/image.jpg'
#
# # Copy the image
# shutil.copy(source_path, destination_path)
#
# # Optional: Remove the original image to mimic 'moving'
# os.remove(source_path)
rsna_csv = r'D:\Breast_ROI\Metadata\RSNA.csv'
vindir_csv = r'D:\Breast_ROI\Metadata\Vindir.csv'
NlHealth_csv = r'D:\Breast_ROI\Metadata\NL_Health.csv'

rsna_df = pd.read_csv(rsna_csv)
vindir_df = pd.read_csv(vindir_csv)
nlhealth_df = pd.read_csv(NlHealth_csv)

def getCancer(subPath, dataName):
    if dataName == 'RSNA':
        for filename in os.listdir(subPath):
            if filename.lower().endswith('.png'):
                image_id = filename.split('_')[1].split('.')[0]
                filtered_df = rsna_df[rsna_df['image_id'] == int(image_id)]
                if not filtered_df.empty:
                    cancer = filtered_df['cancer'].iloc[0]
                    return os.path.join(subPath, filename), cancer
    elif dataName == 'Vindir':
        try:
            for filename in os.listdir(subPath):
                if filename.lower().endswith('.png'):
                    image_id = filename.split('_')[1].split('.')[0]
                    filtered_df = vindir_df[vindir_df['image_id'] == image_id]
                    if not filtered_df.empty:
                        cancer = filtered_df['breast_birads'].iloc[0]
                        if (cancer == 'BI-RADS 5' or cancer == 'BI-RADS 4'):
                            index = 1
                        else:
                            index = 0

                        return os.path.join(subPath, filename), index
        except:
            return False, False
    else:
        try:
            for filename in os.listdir(subPath):
                if filename.lower().endswith('.png'):
                    image_id = filename.split('_')[1].split('.')[0]
                    if len(image_id) == 4:
                        return os.path.join(subPath, filename), 0
                    else:
                        patient_id = subPath.split('\\')[-2]
                        filtered_df = nlhealth_df[(nlhealth_df['image_id'] == image_id) & (nlhealth_df['patient_id'] == patient_id)]
                    if not filtered_df.empty:
                        cancer = filtered_df['cancer'].iloc[0]
                        return os.path.join(subPath, filename), cancer
        except:
            return False, False



save_folder = r'C:\Users\Woody\Desktop\All_Positive_Cases'
for dataset in ['Vindir']:
    datasetPath = os.path.join(r'D:\Breast_ROI', dataset)
    for item in os.listdir(datasetPath):
        folderPath = os.path.join(datasetPath, item)

        CC_folder = ['CC_left', 'CC_right']
        MLO_folder = ['MLO_left', 'MLO_right']
        images = []
        labels = []
        for catogry in [CC_folder,MLO_folder]:
            images = []
            labels = []
            for subfolder in catogry:
                subPath = os.path.join(folderPath, subfolder)
                image_path, label = getCancer(subPath, dataset)
                images.append(image_path)
                labels.append(label)

            if 1 in labels:
                for index in range(2):
                    subfolder = catogry[index]
                    image_path = images[index]
                    if image_path != False:
                        newPath = os.path.join(save_folder, subfolder)
                        image_name = os.path.basename(image_path)
                        savePath = os.path.join(newPath, image_name)

                        shutil.copy(image_path, savePath)



