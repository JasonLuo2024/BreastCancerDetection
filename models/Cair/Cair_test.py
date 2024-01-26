import sys
specific_folder_path = r'/gpfs/home/hluo/anaconda3/envs/torch/lib'
sys.path.append(specific_folder_path)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from tqdm.auto import tqdm
import torch.nn as nn
from fourViewModel import fourViewModel
from fourViewDataset import trainDataset,testDataset

LEARNING_RATE = 0.00015
avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0

def getAverage(CC_left,CC_right,MLO_left,MLO_right):
    average_left = (F.softmax(CC_left, dim=1) + F.softmax(MLO_left, dim=1)) / 2
    average_right = (F.softmax(CC_right, dim=1) + F.softmax(MLO_right, dim=1)) / 2
    return torch.cat((average_left, average_right, average_left, average_right), dim=0)


def main():
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

    train_dataset = trainDataset(r'/gpfs/home/hluo/Honros/Dataset/Vindir_ROI_2', transform=train_transform)
    test_dataset = testDataset(r'/gpfs/home/hluo/Honros/Dataset/RSNA', transform=test_transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(fourViewModel()).to(device)
    else:
        model = fourViewModel().to(device)

    weights = torch.tensor([1.0, 49.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)



    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=20)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=20)

    num_epochs = 40
    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        try:
            for CC_L, CC_R, MLO_L, MLO_R, CC_L_label, CC_R_label, MLO_L_label, MLO_R_label in tqdm(train_dataloader):
                model.train()
                CC_L = CC_L.to(device)
                CC_R = CC_R.to(device)
                MLO_L = MLO_L.to(device)
                MLO_R = MLO_R.to(device)

                labels = torch.cat((CC_L_label, CC_R_label, MLO_L_label, MLO_R_label), dim=0).to(device)

                CC_left,CC_right,MLO_left,MLO_right = model(CC_L, CC_R, MLO_L, MLO_R)

                outputs = torch.cat((CC_left, CC_right, MLO_left, MLO_right), dim=0)

                _, pred = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                average = getAverage(CC_left,CC_right,MLO_left,MLO_right)

                _, preds = torch.max(average, 1)

                optimizer.zero_grad()
                #
                loss.backward()
                #
                optimizer.step()

                running_loss += loss.item() * (CC_L.size(0) * 4 )
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / (len(train_dataset) * 4)
            epoch_acc = running_corrects.double() / (len(train_dataset) * 4)

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for CC_L, CC_R, MLO_L, MLO_R, CC_L_label, CC_R_label, MLO_L_label, MLO_R_label in tqdm(test_dataloader):
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
            recall = recall_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            print(f"\nF1-Score: {f1:.5f} | Recall: {recall:.2f}")
            print(f"\nSpecificity: {specificity:.5f} | sensitivity: {sensitivity:.5f} | Accuracy: {accuracy:.2f}%\n")


            output_file = r'/gpfs/home/hluo/Honros/result/batch_size 8/'+str(epoch)+ 'metrics.txt'
            with open(output_file, "w") as file:
                file.write(f'Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}\n')
                file.write(f"F1-Score: {f1:.5f} | Recall: {recall:.2f}\n")
                file.write(f"Specificity: {specificity:.5f} | Sensitivity: {sensitivity:.5f} | Accuracy: {accuracy:.2f}%\n")


        except Exception as e:
            output_file = r'/gpfs/home/hluo/Honros/result/' + str(epoch) + 'metrics.txt'
            with open(output_file, "w") as file:
                file.write(f'exception {e}')



if __name__ == '__main__':
    main()