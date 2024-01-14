__author__ = "JasonLuo"
import sys
specific_folder_path = r'/gpfs/home/hluo/anaconda3/envs/CNNS/lib'
sys.path.append(specific_folder_path)
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import torchvision.models as models
from tqdm.auto import tqdm
import torch.nn as nn
import torchvision.ops as ops
from models import twoViewChannel


rsna_csv = r'/gpfs/home/hluo/Honros/Dataset/Metadata/RSNA.csv'
rsna_df = pd.read_csv(rsna_csv)

def getLabels(label,batch_size):
    labels = []
    for i in range(batch_size):
        for element in label:
            labels.append(element)
    final_tensor = torch.tensor(labels)
    return final_tensor
def getCancer(subPath, dataName):
    if dataName == 'RSNA':
        for filename in os.listdir(subPath):
            if filename.lower().endswith('.png'):
                image_id = filename.split('_')[1].split('.')[0]
                filtered_df = rsna_df[rsna_df['image_id'] == int(image_id)]
                if not filtered_df.empty:
                    cancer = filtered_df['cancer'].iloc[0]
                    return os.path.join(subPath, filename), cancer


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

        for item in os.listdir(root_dir):
            folderPath = os.path.join(root_dir, item)

            for subfolder in ['CC_left', 'CC_right', 'MLO_left', 'MLO_right']:
                subPath = os.path.join(folderPath, subfolder)
                image_path, label = getCancer(subPath, 'RSNA')
                if image_path != False:
                    if (subfolder == 'CC_left' or subfolder == 'CC_right'):
                        self.CC_list.append(image_path)
                        self.CC_label.append(label)
                    else:
                        self.MLO_list.append(image_path)
                        self.MLO_label.append(label)

        for subpaths, dirs, files in os.walk(r'/gpfs/home/hluo/Honros/SingleModel/Dataset/Positive'):
            for file in files:
                file_path = os.path.join(subpaths, file)
                self.MLO_list.append(file_path)
                self.CC_list.append(file_path)
                self.CC_label.append(1)
                self.MLO_label.append(1)
                label_1 += 1

        print('Negative and Positive cases are', label_0, label_1)
        print(min(len(self.CC_list),len(self.MLO_list)))
        print('image_load succeed')

    def __len__(self):
        return min(len(self.CC_list),len(self.MLO_list))

    def __getitem__(self, idx):
        CC = self.transform(Image.open(self.CC_list[idx]).convert('RGB'))
        MLO = self.transform(Image.open(self.MLO_list[idx]).convert('RGB'))

        CC_label = self.CC_label[idx]
        MLO_label = self.MLO_label[idx]

        return CC, MLO, CC_label, MLO_label


class testDataset(Dataset):
    def __init__(self, root_dir, transform=None):

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
                image_path, label = getCancer(subPath, 'RSNA')
                if image_path != False:
                    if (subfolder == 'CC_left' or subfolder == 'CC_right'):
                        self.CC_list.append(image_path)
                        self.CC_label.append(label)
                    else:
                        self.MLO_list.append(image_path)
                        self.MLO_label.append(label)

        print('Negative and Positive cases are', label_0, label_1)
        print(len(self.CC_list))
        print('image_load succeed')

    def __len__(self):
        return min(len(self.CC_list),len(self.MLO_list))

    def __getitem__(self, idx):
        CC = self.transform(Image.open(self.CC_list[idx]).convert('RGB'))
        MLO = self.transform(Image.open(self.MLO_list[idx]).convert('RGB'))

        CC_label = self.CC_label[idx]
        MLO_label = self.MLO_label[idx]

        return CC, MLO, CC_label, MLO_label



LEARNING_RATE = 0.0001


avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0


def getAverage(CC_left, CC_right, MLO_left, MLO_right):
    average_left = (F.softmax(CC_left, dim=1) + F.softmax(MLO_left, dim=1)) / 2
    average_right = (F.softmax(CC_right, dim=1) + F.softmax(MLO_right, dim=1)) / 2
    return torch.cat((average_left, average_right, average_left, average_right), dim=0)


def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(twoViewChannel()).to(device)
    else:
        model = twoViewChannel().to(device)


    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomRotation((-25, 25)),  # This will rotate between -15 and 15 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = trainDataset(r'/gpfs/home/hluo/Honros/SingleModel/Dataset/RSNA', transform=train_transform)
    test_dataset = testDataset(r'/gpfs/home/hluo/Honros/SingleModel/Dataset/RSNA_Test', transform=test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=7, shuffle=True, num_workers=40)
    test_dataloader = DataLoader(test_dataset, batch_size=7, shuffle=False, num_workers=40,drop_last=True)

    num_epochs = 1000
    try:
        for epoch in tqdm(range(num_epochs)):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0

            for CC,MLO, label1,label2 in tqdm(train_dataloader):
                model.train()
                labels = torch.cat((label1,label2),dim=0).to(device)
                CC = CC.to(device)
                MLO = MLO.to(device)

                output = model(CC,MLO)



                probabilities = torch.sigmoid(output[:, 0])
                preds = probabilities >= 0.3
                preds = preds.int()

                loss = criterion(output, labels)

                optimizer.zero_grad()
                #
                loss.backward()
                #
                optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
            #
            epoch_loss = running_loss / (len(train_dataset))
            epoch_acc = running_corrects.double() / (len(train_dataset))


            y_true = []
            y_pred = []
            with torch.no_grad():
                model.eval()

                for image, label in tqdm(test_dataloader):
                    model.train()
                    labels = torch.cat((label1, label2), dim=0).to(device)
                    CC = CC.to(device)
                    MLO = MLO.to(device)

                    output = model(CC, MLO)

                    labels = label.to(device)

                    output = model(image)
                    probabilities = torch.sigmoid(output[:, 0])
                    preds = probabilities >= 0.3
                    preds = preds.int()

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            f1 = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            auc = roc_auc_score(y_true, y_pred)
            output_file = r'/gpfs/home/hluo/Honros/AveragePooling/Result/DualChannel.txt'

            # Open the file in append mode
            with open(output_file, "a") as file:
                file.write(f'Loss: {epoch_loss:.4f} | Acc: {epoch_acc / 6:.4f}\n')
                file.write(f"F1-Score: {f1:.5f} | AUC: {auc:.5f} | Recall: {recall:.2f}\n")
                file.write(
                    f"Specificity: {specificity:.5f} | Sensitivity: {sensitivity:.5f} | Accuracy: {accuracy:.2f}\n")



    except Exception as e:
        output_file = r'/gpfs/home/hluo/Honros/AveragePooling/Result/errors.txt'
        with open(output_file, "w") as file:
            file.write(f'exception {e}')


if __name__ == '__main__':
    main()