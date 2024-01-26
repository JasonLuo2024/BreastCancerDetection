import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm.auto import tqdm
import torch.nn as nn
from fourViewModel import fourViewModel
from fourViewDataset import trainDataset,testDataset
import mlflow
import torch.nn.functional as F

LEARNING_RATE = 0.001
mlflow.set_tracking_uri('http://127.0.0.1:5000')
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

    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomRotation((-15, 15)),  # This will rotate between -15 and 15 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    train_dataset =  trainDataset(r'D:\Breast_ROI', transform=train_transform)
    test_dataset = testDataset(r'D:\Breast_ROI\NL_Health', transform=test_transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)



    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(fourViewModel()).to(device)
    else:
        model = fourViewModel().to(device)




    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    weights = torch.tensor([1.0, 49.0]).to(device)  # Example weights for two classes

    # Applying these weights to the loss function
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # if dataset is imbalanced -> Adam, otherwise -> SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=6)




    with mlflow.start_run(run_name=run_names):
        num_epochs = 40
        for epoch in tqdm(range(num_epochs)):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0

            for CC_L, CC_R, MLO_L, MLO_R, CC_L_label, CC_R_label,MLO_L_label,MLO_R_label in tqdm(train_dataloader):
                model.train()
                CC_L = CC_L.to(device)
                CC_R = CC_R.to(device)
                MLO_L = MLO_L.to(device)
                MLO_R = MLO_R.to(device)

                labels = torch.cat((CC_L_label, CC_R_label,MLO_L_label,MLO_R_label), dim=0).to(device)

                CC_left,CC_right,MLO_left,MLO_right = model(CC_L, CC_R, MLO_L, MLO_R)

                outputs = torch.cat((CC_left, CC_right, MLO_left, MLO_right), dim=0)

                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs , 1)

                optimizer.zero_grad()
                #
                loss.backward()
                #
                optimizer.step()

                running_loss += loss.item() * (CC_L.size(0) * 4)
                running_corrects += torch.sum(preds == labels.data)
            #
            epoch_loss = running_loss / (len(train_dataset) * 4)
            epoch_acc = running_corrects.double() / (len(train_dataset) * 4)

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
            mlflow.log_metric('loss', epoch_loss, step=epoch)
            mlflow.log_metric('train_accuracy', epoch_acc, step=epoch)
            with torch.no_grad():
                for CC_L, CC_R, MLO_L, MLO_R, CC_L_label, CC_R_label,MLO_L_label,MLO_R_label in tqdm(test_dataloader):
                    CC_L = CC_L.to(device)
                    CC_R = CC_R.to(device)
                    MLO_L = MLO_L.to(device)
                    MLO_R = MLO_R.to(device)

                    labels = torch.cat((CC_L_label, CC_R_label, MLO_L_label, MLO_R_label), dim=0).to(device)

                    CC_left, CC_right, MLO_left, MLO_right = model(CC_L, CC_R, MLO_L, MLO_R)

                    average = getAverage(CC_left,CC_right,MLO_left,MLO_right)

                    _, preds = torch.max(average, 1)

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