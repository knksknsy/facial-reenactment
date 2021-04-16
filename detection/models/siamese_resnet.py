import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from io import StringIO
from torchsummary import summary

from configs.options import Options
from utils.models import init_weights
from ..models.resnet import BasicBlock, ResNet

class SiameseResNet(nn.Module):
    def __init__(self, options: Options, len_feature: int, mask_loss: bool):
        super(SiameseResNet, self).__init__()
        self.options = options
        num_feature = self.options.hidden_layer_num_features

        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], len_feature=len_feature, mask_loss=mask_loss)

        # TODO: add Dropout
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
    def forward_feature(self, x1, x2):
        x1, mask1 = self.resnet(x1)
        x2, mask2 = self.resnet(x2)
        return x1, x2, mask1, mask2


    # Get prediction for single feature vec
    def forward_classification(self, x):
        x, mask = self.resnet(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x, mask


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
    def __init__(self, options: Options, type: str, margin: float):
        super(LossSiamese, self).__init__()
        self.options = options
        self.w_mask = self.options.l_mask
        self.bce_loss = nn.BCELoss()
        self.mask_loss = nn.MSELoss()

        if type == 'contrastive':
            self.loss = ContrastiveLoss(margin)
        elif type == 'triplet':
            self.loss = TripletLoss(margin)

        self.to(self.options.device)


    def forward_feature(self, x1, x2, target, m1, m2, mask1, mask2, real_pair: bool):
        loss_contrastive = self.loss(x1, x2, target)
        loss_mask = self.w_mask * (self.mask_loss(m1, mask1) + self.mask_loss(m2, mask2)) if self.w_mask > 0 else 0

        loss = loss_contrastive + loss_mask

        # LOSSES DICT
        losses = dict({
            'Loss_Contr': loss_contrastive.detach().item(),
            'Loss_Contr_Real': loss_contrastive.detach().item() if real_pair else 0.0,
            'Loss_Contr_Fake': loss_contrastive.detach().item() if not real_pair else 0.0,
            'Loss_Mask': loss_mask.detach().item() if self.w_mask > 0 else 0.0,
            'Loss_Feature': loss.detach().item()
        })

        return loss, losses


    def forward_classification(self, prediction, target, m, mask):
        loss_bce = self.bce_loss(prediction, target)
        loss_mask = self.w_mask * self.mask_loss(m, mask) if self.w_mask > 0 else 0

        loss = loss_bce + loss_mask

        # LOSSES DICT
        losses = dict({
            'Loss_BCE': loss_bce.detach().item(),
            'Loss_Mask': loss_mask.detach().item() if self.w_mask > 0 else 0.0,
            'Loss_Class': loss.detach().item()
        })

        return loss, losses


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, vec1, vec2, label):
        distance = F.mse_loss(vec1, vec2, reduction='none')
        loss = torch.mean(0.5 * (label * distance.pow(2)) + (1 - label) * F.relu(self.margin - distance).pow(2))
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin


    def forward(self, anchor, positive, negative):
        loss = torch.mean(F.relu(F.mse_loss(anchor, positive, reduction='none') - F.mse_loss(anchor, negative, reduction='none') + self.margin))
        return loss
