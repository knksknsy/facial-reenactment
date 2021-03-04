import collections
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, vgg_model, channels):
        super(VGG, self).__init__()
        self.channels = channels
        self.loss_output = collections.namedtuple('loss_output', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        
        # If model is used in DataParallel
        self.vgg_layers = vgg_model.features if hasattr(vgg_model,'features') else vgg_model.module.features

        self.layer_name_mapping = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'
        }

    def forward(self, x):
        if self.channels == 1:
            x = torch.cat((x,)*3, dim=1)
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x

        return self.loss_output(**output)
