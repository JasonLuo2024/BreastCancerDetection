import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torchvision.models as models
from tqdm.auto import tqdm
import torch.nn as nn
import pandas as pd


## the model is built based on nyukat breast_cancer_classifier

class AllViewsGaussianNoise(nn.Module):
    """Add gaussian noise across all 4 views"""

    def __init__(self, gaussian_noise_std):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x):
        self.single_add_gaussian_noise(x)
        return self.single_add_gaussian_noise(x)

    def single_add_gaussian_noise(self, single_view):
        if not self.gaussian_noise_std or not self.training:
            return single_view
        return single_view + single_view.new(single_view.shape).normal_(std=self.gaussian_noise_std)


class AllViewsAvgPool(nn.Module):
    """Average-pool across all 4 views"""

    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        n, c, _, _ = x.size()
        return x.view(n, c, -1).mean(-1)

    @staticmethod
    def single_avg_pool(single_view):
        n, c, _, _ = single_view.size()
        return single_view.view(n, c, -1).mean(-1)


class modifiedDensenet169(nn.Module):
    def __init__(self):
        super(modifiedDensenet169, self).__init__()

        self.model_1 = models.wide_resnet50_2(pretrained=True)
        # self.model_2 = models.densenet169(pretrained=True)

        num_features = self.model_1.fc.in_features

        self.all_views_avg_pool = AllViewsAvgPool()
        self.all_views_gaussian_noise_layer = AllViewsGaussianNoise(0.01)



        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.ReLU(),  # Using a consistent activation function (ReLU)
            nn.Dropout(0.4),  # Slightly reduced dropout
            nn.Linear(512, 2)
        )

        # Applying the same classifier to both models
        self.model_1.classifier = self.classifier
        # self.model_2.classifier = self.classifier

    def forward(self, x1,x2):
        h1 = self.all_views_gaussian_noise_layer.single_add_gaussian_noise(x1)
        # h2 = self.all_views_gaussian_noise_layer.single_add_gaussian_noise(x2)
        result1 = self.model_1(h1)
        # result2 = self.model_2(h2)
        result1 = self.all_views_avg_pool.single_avg_pool(result1)
        # result2 = self.all_views_avg_pool.single_avg_pool(result2)
        output1 = self.classifier(result1)
        # output2 = self.classifier(result2)
        # output = torch.cat((output1, output2), dim=0)

        # Experiment with different ways of merging - here using average

        return output1



