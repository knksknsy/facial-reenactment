import torch
import torch.nn as nn
from torchvision.models import inception_v3

from configs.options import Options

class InceptionNetwork(nn.Module):
    def __init__(self, options: Options, transform_input=True):
        super().__init__()
        self.options = options
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input


    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output


    def forward(self, x):
        if self.options.channels == 1:
            x = torch.cat((x,)*3, dim=1)

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations
