import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import layers as layers


class twoViewChannel(nn.Module):
    def __init__(self):
        super(twoViewChannel, self).__init__()
        self.CC  = modifiedDensenet169()
        self.MLO = modifiedDensenet169()

    def forward(self, CC, MLO):
        output1 = self.CC(CC)
        output2 = self.MLO(MLO)
        output = torch.cat((output1, output2), dim=0)

        return output

class modifiedDensenet169(nn.Module):
    def __init__(self):
        super(modifiedDensenet169, self).__init__()

        self.model_1 = models.densenet201(pretrained=True)

        num_features = self.model_1.classifier.in_features

        self.all_views_avg_pool = layers.AllViewsAvgPool()
        self.all_views_gaussian_noise_layer = layers.AllViewsGaussianNoise(0.01)
        self.output_layer = layers.OutputLayer(1000, (2, 2))

    def forward(self, x1):
        h1 = self.all_views_gaussian_noise_layer.single_add_gaussian_noise(x1)

        result1 = self.model_1(h1)

        result1 = self.all_views_avg_pool.single_avg_pool(result1)

        result1 = self.output_layer(result1)

        return result1


# class LogResenet(nn.Module):
#     def __init__(self):
#         super(LogResenet, self).__init__()
#         self.four_view_resnet = SixViewResNet()
#     def forward(self,x):
#         LogScale = Log.LaplacianOfGaussian(x)
#         output = SixViewResNet(LogScale)
#         return output
#
# class SixViewResNet(nn.Module):
#     def __init__(self):
#         super(SixViewResNet, self).__init__()
#         self.filter_0 = SingleResnet50()
#         self.filter_1 = SingleResnet50()
#         self.filter_2 = SingleResnet50()
#         self.filter_3 = SingleResnet50()
#         self.filter_4 = SingleResnet50()
#         self.filter_5 = SingleResnet50()
#         self.filter_6 = SingleResnet50()
#
#         self.model_dict = {'0': self.filter_0,
#                            '1': self.filter_1,
#                            '2': self.filter_2,
#                            '3': self.filter_3,
#                            '4': self.filter_4,
#                            '5': self.filter_5,
#                            '6': self.filter_6}
#
#
#     def addDevice(self,device):
#         for index, item in enumerate(self.model_dict):
#             self.model_dict[str(index)].to(device)
#
#     def forward(self, x_dict,train):
#         outputs = {}
#         test = []
#         for index, item in enumerate(x_dict):
#             outputs[str(index)] = self.model_dict[str(index)](x_dict[str(index)])
#             test.append(outputs[str(index)])
#         output = torch.cat(test, dim=0)
#
#         if train != True:
#
#             _, output = torch.max(outputs[str(0)], 1)
#             for index, item in enumerate(outputs):
#                 _, scale  = torch.max(outputs[str(index)], 1)
#                 output = torch.bitwise_or(output, scale)
#
#         return output




