# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
"""
Defines layers used in models.py.
"""
import numpy as np
import torch.nn as nn
from torchvision.models.resnet import conv3x3
import torch


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features):
        super(FullyConnectedLayer, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.ReLU(),  # Using a consistent activation function (ReLU)
            nn.Dropout(0.4),  # Slightly reduced dropout
            nn.Linear(512, 2)
        )

    def forward(self, x_dict):
        outputs = []
        for index, item in enumerate(x_dict):
            print(x_dict[str(index)].shape)
            outputs.append(self.classifier(x_dict[str(index)]))
        concatenated_output = torch.cat(outputs, dim=1)
        return concatenated_output

class OutputLayer(nn.Module):
    def __init__(self, in_features, output_shape):
        super(OutputLayer, self).__init__()
        if not isinstance(output_shape, (list, tuple)):
            output_shape = [output_shape]
        self.output_shape = output_shape
        self.flattened_output_shape = int(np.prod(output_shape))
        self.fc_layer = nn.Linear(in_features, self.flattened_output_shape)
        self.fc = nn.Linear(self.flattened_output_shape, 2)

    def forward(self, x):
        h = self.fc_layer(x)
        h = self.fc(h)
        # if len(self.output_shape) > 1:
        #     h = h.view(h.shape[0], *self.output_shape)
        # h = F.log_softmax(h, dim=-1)
        return h




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
        # for index, item in enumerate(x):
        #     # print(x[str(index)].shape)
        #     x[str(index)] = self.single_avg_pool((x[str(index)]))
        return {
            view_name: self.single_avg_pool(view_tensor)
            for view_name, view_tensor in x.items()
        }
        # return x

    @staticmethod
    def single_avg_pool(single_view):
        n, c = single_view.size()
        return single_view.view(n, c, -1).mean(-1)
