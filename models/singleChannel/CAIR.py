__author__ = "JasonLuo"
import sys
specific_folder_path = r'/gpfs/home/hluo/anaconda3/envs/torch/lib'
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
import torchvision


class modifiedDensenet169(nn.Module):
    def __init__(self):
        super(modifiedDensenet169, self).__init__()

        # Using DenseNet169 instead of ResNet50 for potentially better feature extraction
        self.model_1 = models.resnet152(pretrained=True)

        # Freezing the early layers of the model to retain pre-trained features

        # Getting the number of input features for the classifier
        num_features = self.model_1.fc.in_features

        # Simplified classifier with optimized dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.ReLU(),  # Using a consistent activation function (ReLU)
            nn.Dropout(0.4),  # Slightly reduced dropout
            nn.Linear(512, 2)
        )
        # Applying the same classifier to both models
        self.model_1.classifier = self.classifier

    def forward(self, x1):
        output = self.model_1(x1)

        # Experiment with different ways of merging - here using average
        return output



# (43669, 954)
# (11037, 204)
rsna_csv = r'/gpfs/home/hluo/Honros/Dataset/Metadata/RSNA.csv'
rsna_df = pd.read_csv(rsna_csv)

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




        for subpaths, dirs, files in os.walk(r'/gpfs/home/hluo/Honros/SingleModel/Dataset/Positive'):
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


LEARNING_RATE = 0.0001

run_names = "final_test"
avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0



def getAverage(CC_left,CC_right,MLO_left,MLO_right):
    average_left = (F.softmax(CC_left, dim=1) + F.softmax(MLO_left, dim=1)) / 2
    average_right = (F.softmax(CC_right, dim=1) + F.softmax(MLO_right, dim=1)) / 2
    return torch.cat((average_left, average_right, average_left, average_right), dim=0)


def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)



    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(modifiedDensenet169()).to(device)
    else:
        model = modifiedDensenet169().to(device)

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




    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # weights = torch.tensor([1.0, 49.0]).to(device)  # Example weights for two classes

    # Applying these weights to the loss function
    criterion = torchvision.ops.focal_loss()

    # if dataset is imbalanced -> Adam, otherwise -> SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=40)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=20)

    num_epochs = 100
    try:
        for epoch in tqdm(range(num_epochs)):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0

            for image, label in tqdm(train_dataloader):
                model.train()
                image = image.to(device)
                labels = label.to(device)

                output = model(image)

                _, preds = torch.max(output, 1)

                loss = criterion(output, labels)

                _, preds = torch.max(output, 1)

                optimizer.zero_grad()
                #
                loss.backward()
                #
                optimizer.step()

                running_loss += loss.item() * (image.size(0))
                running_corrects += torch.sum(preds == labels.data)
            #
            epoch_loss = running_loss / (len(train_dataset))
            epoch_acc = running_corrects.double() / (len(train_dataset))

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            model.eval()
            y_true = []
            y_pred = []

            with torch.no_grad():
                for image, label in tqdm(test_dataloader):
                    model.train()
                    image = image.to(device)
                    labels = label.to(device)

                    output = model(image)

                    _, preds = torch.max(output, 1)

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            auc = roc_auc_score(y_true, y_pred)
            output_file = r'/gpfs/home/hluo/Honros/SingleModel/result/new_synthetic1.txt'

            # Open the file in append mode
            with open(output_file, "a") as file:
                file.write(f'Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}\n')
                file.write(f"F1-Score: {f1:.5f} | AUC: {auc:.5f} | Recall: {recall:.2f}\n")
                file.write(
                    f"Specificity: {specificity:.5f} | Sensitivity: {sensitivity:.5f} | Accuracy: {accuracy:.2f}\n")



    except Exception as e:
        output_file = r'/gpfs/home/hluo/Honros/SingleModel/result/synthetic.txt'
        with open(output_file, "w") as file:
            file.write(f'exception {e}')






if __name__ == '__main__':
    main()