__author__ = "JasonLuo"
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm.auto import tqdm
import torch.nn as nn
from twoViewModel import modifiedDensenet169
from twoViewDataset import trainDataset,testDataset
import mlflow
import torch.nn.functional as F
import torchvision

LEARNING_RATE = 0.002
mlflow.set_tracking_uri('http://127.0.0.1:5000')
run_names = "final_test"
avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0

def getAverage(CC_left,CC_right,MLO_left,MLO_right):
    average_left = (F.softmax(CC_left, dim=1) + F.softmax(MLO_left, dim=1)) / 2
    average_right = (F.softmax(CC_right, dim=1) + F.softmax(MLO_right, dim=1)) / 2
    return torch.cat((average_left, average_right, average_left, average_right), dim=0)

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0.7, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = (1 - pt) ** self.gamma * BCE_loss
#
#         if self.reduction == 'mean':
#             return torch.mean(F_loss)
#         elif self.reduction == 'sum':
#             return torch.sum(F_loss)
#         else:
#             return F_loss

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

    train_dataset = trainDataset(r'D:\Breast_ROI', transform=train_transform)
    test_dataset = testDataset(r'D:\Breast_ROI\NL_Health_test', transform=test_transform)


    criterion = torch.nn.CrossEntropyLoss()

    # if dataset is imbalanced -> Adam, otherwise -> SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=6)




    with mlflow.start_run(run_name=run_names):
        num_epochs = 40
        for epoch in tqdm(range(num_epochs)):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0

            for CC,MLO,CC_label,MLO_label in tqdm(train_dataloader):
                model.train()
                CC = CC.to(device)
                MLO = MLO.to(device)

                labels = torch.cat((CC_label, MLO_label), dim=0).to(device)

                output = model(CC, MLO)

                _, preds = torch.max(output, 1)

                loss = criterion(output, labels)

                _, preds = torch.max(output , 1)

                optimizer.zero_grad()
                #
                loss.backward()
                #
                optimizer.step()

                running_loss += loss.item() * (CC.size(0) * 2)
                running_corrects += torch.sum(preds == labels.data)
            #
            epoch_loss = running_loss / (len(train_dataset)*2)
            epoch_acc = running_corrects.double() / (len(train_dataset) * 2)

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            model.eval()  # Set the model to evaluation mode
            # #
            y_true = []
            y_pred = []

            # test_labels = []
            # predicted_labels = []
            # patient_ID = []
            # image_path = []

            # train_csv = r'/gpfs/home/hluo/Dataset/Vindr/data/breast-level_annotations.csv'
            # # # # evaluate the result at each epoch
            # mlflow.log_metric('loss', epoch_loss, step=epoch)
            # mlflow.log_metric('train_accuracy', epoch_acc, step=epoch)
            with torch.no_grad():
                for CC,MLO,CC_label,MLO_label in tqdm(test_dataloader):
                    CC = CC.to(device)
                    MLO = MLO.to(device)

                    labels = torch.cat((CC_label,MLO_label), dim=0).to(device)

                    output = model(CC,MLO)

                    _, preds = torch.max(output, 1)

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            # df = pd.DataFrame({
            #             'Test Labels': test_labels,
            #             'Predicted Labels': predicted_labels,
            #             'Patient ID': patient_ID,
            #             'Image Path': image_path
            # })
            # df.to_csv(r'/gpfs/home/hluo/Dataset/Vindr/SingleCNN/'+'output_' + str(epoch) + '.csv', index=False)
            # Calculate evaluation metrics
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)


            # output_file = r'/gpfs/home/hluo/Dataset/Vindr/SingleCNN/'+str(epoch)+ 'metrics.txt'
            # with open(output_file, "w") as file:
            #     file.write(f'Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}\n')
            #     file.write(f"F1-Score: {f1:.5f} | Recall: {recall:.2f}\n")
            #     file.write(f"Specificity: {specificity:.5f} | Sensitivity: {sensitivity:.5f} | Accuracy: {accuracy:.2f}%\n")

            print(f"\nF1-Score: {f1:.5f} | Recall: {recall:.2f}")
            print(f"\nSpecificity: {specificity:.5f} | sensitivity: {sensitivity:.5f} | Accuracy: {accuracy:.2f}\n")
            mlflow.log_metric('f1', f1, step=epoch)
            mlflow.log_metric('precision', precision, step=epoch)
            mlflow.log_metric('recall', recall, step=epoch)
            mlflow.log_metric('specificity', specificity, step=epoch)
            mlflow.log_metric('sensitivity', sensitivity, step=epoch)
            mlflow.log_metric('test_accuracy', accuracy, step=epoch)

        # except Exception as e:
        # output_file = r'/gpfs/home/hluo/Dataset/Vindr/SingleCNN/'+str(epoch)+ 'metrics.txt'
        # with open(output_file, "w") as file:
        #     file.write(f'exception {e}')



if __name__ == '__main__':
    main()