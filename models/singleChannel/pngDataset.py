import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

# (43669, 954)
# (11037, 204)
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


class trainDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.label_list = []

        label_0 = 0
        label_1 = 0

        for item in os.listdir(root_dir):
            folderPath = os.path.join(root_dir, item)

            for subfolder in ['CC_left', 'CC_right', 'MLO_left', 'MLO_right']:
                subPath = os.path.join(folderPath, subfolder)
                image_path, label = getCancer(subPath, 'RSNA')
                if image_path != False:
                    if label == 1:
                        label_1 += 1
                    else:
                        label_0 += 1
                    self.image_list.append(image_path)
                    self.label_list.append(label)




        for subpaths, dirs, files in os.walk(r'D:\Breast_ROI\Positive'):
            for file in files:
                file_path = os.path.join(subpaths,file)
                self.image_list.append(file_path)
                self.label_list.append(1)
                label_1 +=1




        print('Negative and Positive cases are',label_0, label_1)
        print(len(self.image_list))
        print('image_load succeed')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_list[idx]).convert('RGB'))

        label = self.label_list[idx]

        return image,label


class testDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.label_list = []

        label_0 = 0
        label_1 = 0

        for item in os.listdir(root_dir):
            folderPath = os.path.join(root_dir, item)

            for subfolder in ['CC_left', 'CC_right', 'MLO_left', 'MLO_right']:
                subPath = os.path.join(folderPath, subfolder)
                image_path, label = getCancer(subPath, 'RSNA')
                if image_path != False:
                    if label == 1:
                        label_1 += 1
                    else:
                        label_0 += 1
                    self.image_list.append(image_path)
                    self.label_list.append(label)

        print('Negative and Positive cases are',label_0, label_1)
        print(len(self.image_list))
        print('image_load succeed')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.image_list[idx]).convert('RGB'))

        label = self.label_list[idx]

        return image, label


