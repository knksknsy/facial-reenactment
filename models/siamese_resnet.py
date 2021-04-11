from models.components import ConvBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from io import StringIO
from torchsummary import summary

from models.resnet import BasicBlock, ResNet
from configs import Options
from models.utils import init_weights

class SiameseResNet(nn.Module):
    def __init__(self, options: Options, len_feature: int):
        super(SiameseResNet, self).__init__()
        self.options = options

        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], len_feature=len_feature)

        # TODO: add to options
        num_feature = 256

        classifier = []
        classifier.append(Unsqueeze(1))
        classifier.append(nn.Conv1d(1, num_feature, kernel_size=3, stride=1, padding=1, bias=False))
        classifier.append(nn.BatchNorm1d(num_feature, affine=True, track_running_stats=True))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(Flatten())
        classifier.append(nn.Linear(len_feature*num_feature, 1))
        self.classifier = nn.Sequential(*classifier)

        # self.classifier = nn.Linear(len_feature, 1)

        self.apply(init_weights)
        self.to(self.options.device)


    # Siamese mode: get 2 feature vecs
    def forward_feats(self, x1, x2):
        x1, _ = self.resnet(x1)
        x2, _ = self.resnet(x2)
        return x1, x2


    # Get prediction for single feature vec
    def forward_preds(self, x):
        x, _ = self.resnet(x)
        x = self.classifier(x)
        x = F.sigmoid(x)
        return x


    def __str__(self):
        old_stdout = sys.stdout
        sys.stdout = new_stdout = StringIO()
        summary(self, input_size=(self.options.channels, self.options.image_size, self.options.image_size), batch_size=self.options.batch_size, device=self.options.device)
        sys.stdout = old_stdout
        return new_stdout.getvalue()


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim


    def forward(self, x):
        return x.unsqueeze(self.dim)


class LossSiamese(nn.Module):
    def __init__(self, options: Options, margin: float):
        super(LossSiamese, self).__init__()
        self.options = options
        self.contrastive_loss = ContrastiveLoss(margin)
        self.bce_loss = nn.BCELoss()

        self.to(self.options.device)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, vec1, vec2, label):
        distance = F.mse_loss(vec1, vec2, reduction='none')
        loss = torch.mean(0.5 * (label * distance.pow(2)) + (1 - label) * F.relu(self.margin - distance).pow(2))
        return loss
