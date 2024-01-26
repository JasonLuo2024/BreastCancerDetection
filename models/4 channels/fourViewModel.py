import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torchvision.models as models
from tqdm.auto import tqdm
import torch.nn as nn
import pandas as pd

## the model is built based on nyukat breast_cancer_classifier

class modifiedDensennet169(nn.Module):
        def __init__(self):
            super(modifiedDensennet169, self).__init__()
            self.model_1 = models.densenet169(pretrained=True)
            self.num_features = self.model_1.classifier.in_features
            self.model_2 = models.densenet169(pretrained=True)
            
            ##increased the depth of the network by adding more linear layers.
            #introduced batch normalization after each fully connected layer. This can improve the convergence and generalization of the network.
            #ncorporated a mix of activation functions, including ReLU, LeakyReLU, and PReLU to introduce some non-linearity variety.
            #adjusted the dropout rates across layers, gradually decreasing it as we go deeper to retain more information in the latter layers.
            self.classifier = nn.Sequential(
                nn.Linear(self.num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),

                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.PReLU(),
                nn.Dropout(0.2),

                nn.Linear(128, 2)
        )
            ## these two share the same classifier
            self.model_1.classifier = self.classifier
            self.model_2.classifier = self.classifier

        def forward(self, x1,x2):
            left = self.model_1(x1) #CC_Left  -> the apply softmax
            right = self.model_2(x2) #CC_Right  -> the apply softmax
            # output = torch.cat((left, right), dim=0)
            return left, right


class fourViewModel(nn.Module):
     def __init__(self):
            super(fourViewModel, self).__init__()
            self.CC =  modifiedDensennet169()
            self.MLO = modifiedDensennet169()

## calculate the average from  CC and MLO or
     def forward(self, CC_L,CC_R,MLO_L,MLO_R):
            CC_left, CC_right = self.CC(CC_L,CC_R)
            MLO_left, MLO_right = self.MLO(MLO_L,MLO_R)
            return CC_left,CC_right,MLO_left,MLO_right

           