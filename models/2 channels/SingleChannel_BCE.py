__author__ = "JasonLuo"
__author__ = "JasonLuo"
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
from Models import modifiedDensenet169,Densennet169
import mlflow
run_names = 'SingleResnet50_50_BCE_30'


mlflow.set_tracking_uri('http://127.0.0.1:5000')
vindir_csv = r'D:\Breast_ROI\Metadata\Vindir.csv'
vindir_df = pd.read_csv(vindir_csv)

def getCancer(image):
    try:
        image_id = image.split('_')[1].split('.')[0]
        filtered_df = vindir_df[vindir_df['image_id'] == image_id]
        if not filtered_df.empty:
            cancer = filtered_df['breast_birads'].iloc[0]

            viewPosition = filtered_df['view_position'].iloc[0]

            if (cancer == 'BI-RADS 5' or cancer == 'BI-RADS 4'):
                index = 1
            else:
                index = 0

            return index,viewPosition


    except:
        return False, False


class VindirDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.label_list = []

        label_0 = 0
        label_1 = 0

        for subpath, dirs, files in os.walk(root_dir):
            for file in files:
                image_path = os.path.join(subpath, file)
                index, viewPosition = getCancer(file)
                self.image_list.append(image_path)
                self.label_list.append(index)
                if index == 1:
                    label_1 +=1
        label_0 = len(self.image_list)- label_1


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

avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0




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
        transforms.Resize((1024, 512)),
        transforms.RandomRotation((-25, 25)),  # This will rotate between -15 and 15 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
    transforms.Resize((1024, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    train_dataset = VindirDataset(r'D:\thesis\train', transform=train_transform)
    test_dataset = VindirDataset(r'D:\thesis\test', transform=test_transform)



    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=10)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=10)

    num_epochs = 40
    with mlflow.start_run(run_name=run_names):
        try:
            for epoch in tqdm(range(num_epochs)):
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print('-' * 10)

                running_loss = 0.0
                running_corrects = 0

                for image, label in tqdm(train_dataloader):
                    model.train()
                    labels = label.to(device)
                    image = image.to(device)

                    output = model(image)

                    loss = criterion(output[:, 1], labels.float())

                    optimizer.zero_grad()
                    #
                    loss.backward()
                    #
                    optimizer.step()

                    _, preds = torch.max(output, 1)

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
                        labels = label.to(device)
                        image = image.to(device)
                        output = model(image)

                        probabilities = torch.sigmoid(output[:, 1])
                        preds = probabilities >= 0.30
                        preds = preds.int()
                        # _, preds = torch.max(output, 1)

                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(preds.cpu().numpy())

                f1 = f1_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                specificity = tn / (tn + fp)
                sensitivity = tp / (tp + fn)
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                auc = roc_auc_score(y_true, y_pred)
                print(f'Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}\n')
                print(f"F1-Score: {f1:.5f} | AUC: {auc:.5f} | Recall: {recall:.2f}\n")
                print(f"Specificity: {specificity:.5f} | Sensitivity: {sensitivity:.5f} | Accuracy: {accuracy:.2f}\n")

                mlflow.log_metric('f1', f1, step=epoch)
                mlflow.log_metric('AUC', auc, step=epoch)
                # mlflow.log_metric('precision', precision, step=epoch)
                mlflow.log_metric('recall', recall, step=epoch)
                mlflow.log_metric('specificity', specificity, step=epoch)
                mlflow.log_metric('sensitivity', sensitivity, step=epoch)
                mlflow.log_metric('test_accuracy', accuracy, step=epoch)
                mlflow.log_metric('loss', epoch_loss, step=epoch)
                mlflow.log_metric('train_accuracy', epoch_acc, step=epoch)




        except Exception as e:
            print(f'exception {e}')







if __name__ == '__main__':
    main()