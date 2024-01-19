import os
import shutil

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


class trainDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.CC_list = []
        self.MLO_list = []
        self.CC_label = []
        self.MLO_label = []

        label_0 = 0
        label_1 = 0

        for dataset in ['Vindir']:
            datasetPath = os.path.join(root_dir,dataset)
            for item in os.listdir(datasetPath):
                folderPath = os.path.join(datasetPath, item)
                image = []
                labels = []

                for subfolder in ['CC_left', 'CC_right', 'MLO_left', 'MLO_right']:
                    subPath = os.path.join(folderPath, subfolder)
                    image_path, label = getCancer(subPath, dataset)
                    if image_path != False:
                        if label == 1:
                            destination = os.path.join(r'C:\Users\Woody\Desktop\Positive\vindir',item)
                            shutil.copytree(folderPath, destination)
                            break
                            # if (subfolder == 'CC_left' or subfolder == 'CC_right'):
                        #        shutil.copy(image_path,r'C:\Users\Woody\Desktop\negative\left')
                        # else:
                        #      shutil.copy(image_path, r'C:\Users\Woody\Desktop\negative\right')
        #                 else:
        #                     label_0 += 1
        #                 if (subfolder == 'CC_left' or subfolder == 'CC_right'):
        #                     self.CC_list.append(image_path)
        #                     self.CC_label.append(label)
        #                 else:
        #                     self.MLO_list.append(image_path)
        #                     self.MLO_label.append(label)
        #
        # print('Negative and Positive cases are',label_0, label_1)
        # print(len(self.CC_list))
        # print('image_load succeed')

    def __len__(self):
        return min(len(self.CC_list),len(self.MLO_list))

    def __getitem__(self, idx):
        CC = self.transform(Image.open(self.CC_list[idx]).convert('RGB'))
        MLO = self.transform(Image.open(self.MLO_list[idx]).convert('RGB'))


        CC_label = self.CC_label[idx]
        MLO_label = self.MLO_label[idx]

        return CC,MLO,CC_label,MLO_label


class testDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.root_dir = root_dir
        self.transform = transform
        self.CC_list = []
        self.MLO_list = []
        self.CC_label = []
        self.MLO_label = []
        label_0 = 0
        label_1 = 0
        for item in os.listdir(root_dir):
            folderPath = os.path.join(root_dir, item)


            for subfolder in ['CC_left', 'CC_right', 'MLO_left', 'MLO_right']:
                subPath = os.path.join(folderPath, subfolder)
                image_path, label = getCancer(subPath, 'nlhealth')
                if image_path != False:
                    if label == 1:
                        label_1+=1
                    else:
                        label_0+=1
                    if (subfolder == 'CC_left' or subfolder == 'CC_right'):
                        self.CC_list.append(image_path)
                        self.CC_label.append(label)
                    else:
                        self.MLO_list.append(image_path)
                        self.MLO_label.append(label)

        print('Negative and Positive cases are',label_0, label_1)
        print(len(self.CC_list))
        print('image_load succeed')

    def __len__(self):
        return min(len(self.CC_list),len(self.MLO_list))

    def __getitem__(self, idx):
        CC = self.transform(Image.open(self.CC_list[idx]).convert('RGB'))
        MLO = self.transform(Image.open(self.MLO_list[idx]).convert('RGB'))


        CC_label = self.CC_label[idx]
        MLO_label = self.MLO_label[idx]

        return CC,MLO,CC_label,MLO_label


train_dataset = trainDataset(r'D:\Breast_ROI', transform=None)
