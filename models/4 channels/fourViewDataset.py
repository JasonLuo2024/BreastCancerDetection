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
                        index = 0 if cancer != 'BI-RADS 5' else 1
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

        self.CC_left_list = []
        self.CC_Right_list = []
        self.MLO_left_list = []
        self.MLO_Right_list = []
        self.CC_left_label = []
        self.CC_Right_label = []
        self.MLO_left_label = []
        self.MLO_Right_label = []

        for dataset in ['RSNA','Vindir']:
            datasetPath = os.path.join(root_dir,dataset)
            for item in os.listdir(datasetPath):
                folderPath = os.path.join(datasetPath, item)
                image = []
                labels = []

                for subfolder in ['CC_left', 'CC_right', 'MLO_left', 'MLO_right']:
                    subPath = os.path.join(folderPath, subfolder)
                    image_path, label = getCancer(subPath, dataset)
                    if image_path == False: break
                    image.append(image_path)
                    labels.append(label)

                if len(image) == 4:
                    self.CC_left_list.append(image[0])
                    self.CC_left_label.append(labels[0])
                    self.CC_Right_list.append(image[1])
                    self.CC_Right_label.append(labels[1])
                    self.MLO_left_list.append(image[2])
                    self.MLO_left_label.append(labels[2])
                    self.MLO_Right_list.append(image[3])
                    self.MLO_Right_label.append(labels[3])


        print('image_load succeed')

    def __len__(self):
        return len(self.CC_left_list)

    def __getitem__(self, idx):
        CC_L = self.transform(Image.open(self.CC_left_list[idx]).convert('RGB'))
        CC_R = self.transform(Image.open(self.CC_Right_list[idx]).convert('RGB'))
        MLO_L = self.transform(Image.open(self.CC_left_list[idx]).convert('RGB'))
        MLO_R = self.transform(Image.open(self.CC_Right_list[idx]).convert('RGB'))

        CC_L_label = self.CC_left_label[idx]
        CC_R_label = self.CC_Right_label[idx]
        MLO_L_label = self.CC_left_label[idx]
        MLO_R_label = self.CC_Right_label[idx]

        return CC_L, CC_R, MLO_L, MLO_R, CC_L_label, CC_R_label, MLO_L_label, MLO_R_label


class testDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.CC_left_list = []
        self.CC_Right_list = []
        self.MLO_left_list = []
        self.MLO_Right_list = []
        self.CC_left_label = []
        self.CC_Right_label = []
        self.MLO_left_label = []
        self.MLO_Right_label = []

        for item in os.listdir(root_dir):
            folderPath = os.path.join(root_dir, item)
            image = []
            labels = []
            for subfolder in ['CC_left', 'CC_right', 'MLO_left', 'MLO_right']:
                subPath = os.path.join(folderPath, subfolder)
                image_path, label = getCancer(subPath, 'nlhealth')
                image.append(image_path)
                labels.append(label)
                if image_path == False: break

            if len(image) == 4:
                self.CC_left_list.append(image[0])
                self.CC_left_label.append(labels[0])
                self.CC_Right_list.append(image[1])
                self.CC_Right_label.append(labels[1])
                self.MLO_left_list.append(image[2])
                self.MLO_left_label.append(labels[2])
                self.MLO_Right_list.append(image[3])
                self.MLO_Right_label.append(labels[3])

        print(len(self.CC_left_list))
        print('image_load succeed')

    def __len__(self):
        return len(self.CC_left_list)

    def __getitem__(self, idx):
        CC_L = self.transform(Image.open(self.CC_left_list[idx]).convert('RGB'))
        CC_R = self.transform(Image.open(self.CC_Right_list[idx]).convert('RGB'))
        MLO_L = self.transform(Image.open(self.CC_left_list[idx]).convert('RGB'))
        MLO_R = self.transform(Image.open(self.CC_Right_list[idx]).convert('RGB'))

        CC_L_label = self.CC_left_label[idx]
        CC_R_label = self.CC_Right_label[idx]
        MLO_L_label = self.CC_left_label[idx]
        MLO_R_label = self.CC_Right_label[idx]

        return CC_L, CC_R, MLO_L, MLO_R, CC_L_label, CC_R_label, MLO_L_label, MLO_R_label


