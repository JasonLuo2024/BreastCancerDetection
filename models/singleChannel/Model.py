__author__ = "JasonLuo"
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torchvision.models as models
from tqdm.auto import tqdm
import torch.nn as nn
import pandas as pd
class modifiedDensenet169(nn.Module):
    def __init__(self):
        super(modifiedDensenet169, self).__init__()

        # Using DenseNet169 instead of ResNet50 for potentially better feature extraction
        self.model_1 = models.densenet169(pretrained=True)
        self.model_2 = models.densenet169(pretrained=True)

        # Freezing the early layers of the model to retain pre-trained features
        for param in self.model_1.parameters():
            param.requires_grad = False
        for param in self.model_2.parameters():
            param.requires_grad = False

        # Getting the number of input features for the classifier
        num_features = self.model_1.classifier.in_features

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
        self.model_2.classifier = self.classifier

    def forward(self, x1, x2):
        left = self.model_1(x1)
        right = self.model_2(x2)

        # Experiment with different ways of merging - here using average
        output = torch.cat((left, right), dim=0)
        return output