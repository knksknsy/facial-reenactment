import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
import numpy as np

from configs.options import Options
from dataset.utils import denormalize

class VGG16(nn.Module):
    def __init__(self, options: Options, requires_grad=False):
        super(VGG16, self).__init__()
        self.options = options
        vgg_model = vgg16(pretrained=True)
        
        # If model is used in DataParallel
        vgg_features = vgg_model.features if hasattr(vgg_model,'features') else vgg_model.module.features

        self.relu_count = 4
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(4,9):
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(9,16):
            self.slice3.add_module(str(x), vgg_features[x])
        for x in range(16,23):
            self.slice4.add_module(str(x), vgg_features[x])

        self.mean = nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))), requires_grad=False)
        self.std = nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))), requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.options.channels == 1:
            x = torch.cat((x,)*3, dim=1)

        # Denormalize input to range [0;1]
        x = denormalize(x, mean=self.options.normalize[0], std=self.options.normalize[1])
        # Normalize input to vgg16
        x = (x - self.mean) / self.std

        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        out = [h_relu1, h_relu2, h_relu3, h_relu4]
        return out
