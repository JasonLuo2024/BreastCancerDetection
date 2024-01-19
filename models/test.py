__author__ = "JasonLuo"

import sys

# Add the path to the specific folder containing Python packages
specific_folder_path = r'/gpfs/home/phajishafiez/anaconda3/envs/myenv/lib'
sys.path.append(specific_folder_path)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision import models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Define the file path where you want to save the printed content
file_path = r'/gpfs/home/phajishafiez/datasets/Only_PNGs/Kaggla dataset_png/output16bits_baseline_lr01.txt'


class RightBreastDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = []
        self.label_list = []

        # Define a dictionary to map class names (normal and abnormal) to labels (0 and 1)
        class_to_label = {'normal': 0, 'abnormal': 1}

        # Walk through the root directory
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)

            # Only consider directories (ignore files)
            if not os.path.isdir(class_dir):
                continue

            # Get the label for this class
            label = class_to_label[class_name]

            # Iterate through patient folders
            for patient_folder in os.listdir(class_dir):
                patient_dir = os.path.join(class_dir, patient_folder)

                # Get all the images in the patient directory
                image_files = [os.path.join(patient_dir, image) for image in os.listdir(patient_dir)]

                if image_files:
                    # Add all image paths to the list
                    self.image_list.extend(image_files)
                    # Add corresponding labels
                    self.label_list.extend([label] * len(image_files))  # Add corresponding labels

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx]).convert('L')

        if self.transform:
            img = self.transform(img)

        label = self.label_list[idx]
        return img, label


# Define a transformation for the images (resize, grayscale, and normalize)
transform = transforms.Compose([
    transforms.Resize((1024, 512)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])


def load_right_breast_data(train_dir, test_dir):
    train_dataset = RightBreastDataset(root_dir=train_dir, transform=transform)
    test_dataset = RightBreastDataset(root_dir=test_dir, transform=transform)

    return train_dataset, test_dataset


# Define a function to train and evaluate the model with late fusion
def train_and_evaluate_model(train_dataset, model, num_epochs, save_model):
    # Initialize the DataLoader for train and test datasets
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)

    # Use CUDA if available
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).to(device)

    else:
        print(device)
        model = model.to(device)

    print("First few labels:")
    print(train_dataset.label_list[:2000])

    # Training CC loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(train_dataloader, desc=f"Training {save_model} Epoch {epoch + 1}/{num_epochs}", ncols=100) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                pbar.set_postfix(loss=f"{running_loss / (pbar.n + 1):.4f}")

        print(f"Training Epoch {epoch + 1}/{num_epochs} Loss: {running_loss / len(train_dataloader):.4f}")
        with open(file_path, 'a') as file:
            file.write(f"Training Epoch {epoch + 1}/{num_epochs} Loss: {running_loss / len(train_dataloader):.4f}\n")

    # Save the model
    model_save_path = r'/gpfs/home/phajishafiez/datasets/Only_PNGs/Kaggla dataset_png/model_stride2_' + save_model + '.pth'

    torch.save(model.module.state_dict(), model_save_path)

    return model


def evaluate_saved_models(model_CC_path, model, test_dataset):
    # Train and evaluate the model for 1 epoch with late fusion for CC  views
    model_eval = train_and_evaluate_model(train_dataset, model, num_epochs=300, save_model='CC')

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=3)

    # Use CUDA if available
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize evaluation metrics
    tp_CC_total = 0
    fp_CC_total = 0
    tn_CC_total = 0
    fn_CC_total = 0

    threshold = 0.50

    # Initialize lists to store true labels and predicted probabilities
    true_labels_list_CC = []
    predicted_probs_list_CC = []

    with torch.no_grad():
        for (inputs_CC, labels_CC) in tqdm(test_dataloader, total=len(test_dataloader), desc="Evaluating"):
            inputs_CC, labels_CC = inputs_CC.to(device), labels_CC.to(device)

            # Evaluate with CC model
            outputs = model_eval(inputs_CC)
            prob = torch.softmax(outputs, dim=1)

            # calculating the metrics for AUC -Roc for CC view
            true_labels_list_CC.extend(labels_CC.cpu().numpy())
            predicted_probs_list_CC.extend(prob[:, 1].cpu().numpy())

            model_condition = (prob[:, 1] > threshold)

            predicted_model = model_condition.type(torch.float)

            # Calculate TP, FP, TN, FN for CC, MLO, and fused views
            tp = ((predicted_model == 1) & (labels_CC == 1)).sum().item()
            fp = ((predicted_model == 1) & (labels_CC == 0)).sum().item()
            tn = ((predicted_model == 0) & (labels_CC == 0)).sum().item()
            fn = ((predicted_model == 0) & (labels_CC == 1)).sum().item()

            # Accumulate TP, FP, TN, FN values for cc
            tp_CC_total += tp
            fp_CC_total += fp
            tn_CC_total += tn
            fn_CC_total += fn
            # Save results to a CSV file for each image
            image_path = test_dataset.image_list[len(predicted_probs_list_CC) - 1]
            results_df = pd.DataFrame({
                'Image_Path': [image_path],
                'Predicted_Label': [predicted_model.item()],
                'Actual_Label': [labels_CC.item()]
            })

            csv_path = r'/gpfs/home/phajishafiez/datasets/Only_PNGs/Kaggla dataset_png/evaluation_results_per_patient.csv'  # Change this path as needed
            if not os.path.exists(csv_path):
                results_df.to_csv(csv_path, index=False)
            else:
                results_df.to_csv(csv_path, mode='a', header=False, index=False)

    fpr_CC, tpr_CC, _ = roc_curve(true_labels_list_CC, predicted_probs_list_CC)

    roc_auc_CC = auc(fpr_CC, tpr_CC)

    precision, recall, _ = precision_recall_curve(true_labels_list_CC, predicted_probs_list_CC)
    auc_prc_CC = auc(recall, precision)

    # Print or return the TP, FP, TN, FN values for CC, MLO, and fused view
    print(f" CC: TP={tp_CC_total}, FP={fp_CC_total}, TN={tn_CC_total}, FN={fn_CC_total}")

    print("************************")
    print(f'AUC-ROC for CC: {roc_auc_CC:.2f}')
    print('\n')
    print(f'AUC-PRC for CC: {auc_prc_CC:.2f}')

    print("************************")
    with open(file_path, 'a') as file:
        file.write(f" CC: TP={tp_CC_total}, FP={fp_CC_total}, TN={tn_CC_total}, FN={fn_CC_total}.\n")

        file.write("************************.\n")
        file.write(f'AUC-ROC for CC: {roc_auc_CC:.2f}\n')
        file.write("************************.\n")
        file.write(f'AUC-PRC for CC: {auc_prc_CC:.2f}\n')

    # Metrics for CC view
    accuracy_CC = (tp_CC_total + tn_CC_total) / (tp_CC_total + fp_CC_total + tn_CC_total + fn_CC_total)
    recall_CC = tp_CC_total / (tp_CC_total + fn_CC_total)
    specificity_CC = tn_CC_total / (tn_CC_total + fp_CC_total)

    # Print or report the calculated metrics
    print("CC View Metrics:")
    print(f"Accuracy: {accuracy_CC:.4f}")
    print(f"Recall (Sensitivity): {recall_CC:.4f}")
    print(f"Specificity: {specificity_CC:.4f}")

    with open(file_path, 'a') as file:
        file.write("CC View Metrics:\n")
        file.write(f"Accuracy: {accuracy_CC:.4f}\n")
        file.write(f"Recall (Sensitivity): {recall_CC:.4f}\n")
        file.write(f"Specificity: {specificity_CC:.4f}\n")
        precision_CC = tp_CC_total / (tp_CC_total + fp_CC_total)
        f1_CC = 2 * (precision_CC * recall_CC) / (precision_CC + recall_CC)
        print(f"F1 Score: {f1_CC:.4f}")
        print("************************")
        file.write(f"F1 Score: {f1_CC:.4f}\n")
        file.write("************************\n")


if __name__ == '__main__':
    # Define the paths to your train and test directories
    train_dir = r'/gpfs/home/phajishafiez/datasets/Only_PNGs/Kaggla dataset_png/train'
    test_dir = r'/gpfs/home/phajishafiez/datasets/Only_PNGs/Kaggla dataset_png/test'

    # initializing the model path to load
    model_CC_path = r'/gpfs/home/phajishafiez/datasets/Only_PNGs/Kaggla dataset_png/model_stride2_CC.pth'

    # Load the datasets for CC and MLO views separately
    train_dataset, test_dataset = load_right_breast_data(train_dir, test_dir)

    # Print the number of images loaded in the train and test datasets for CC and MLO views
    print(f"Number of images in the train dataset : {len(train_dataset)}")
    print(f"Number of images in the test dataset : {len(test_dataset)}")

    # ****************************defining model CC****************************************
    # Define ResNet-50 model
    model = models.resnet50(pretrained=True)
    # Modify the classification head (the final fully connected layer)
    num_ftrs = model.fc.in_features
    print(num_ftrs)
    # Remove the fully connected layer
    model.fc = nn.Identity()

    # Add global average pooling to reduce spatial dimensions to 1x1
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # Add another convolutional layer with num_classes output channels

    num_classes = 2  # Replace 2 with the actual number of classes
    model.classifier = nn.Conv2d(2048, num_classes, kernel_size=1)  # 2048 for ResNet-50

    # Modify the input layer to accept 1 channel (assuming you're working with grayscale images) and set stride to 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.load_state_dict(torch.load(model_CC_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # *****************************evaluating****************************************************************************
    evaluate_saved_models(model_CC_path, model, test_dataset)