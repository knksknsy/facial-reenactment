import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from configs.options import Options
from dataset.utils import denormalize

class LightCNN(nn.Module):
    def __init__(self, options: Options, requires_grad=False):
        super(LightCNN, self).__init__()
        self.options = options
        self.output_count = 2
        layers = [1, 2, 3, 4]
        num_classes=79077

        self.conv1  = mfm(1, 48, 5, 1, 2)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(resblock, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(resblock, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(resblock, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(resblock, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc     = mfm(8*8*128, 256, type=0)
        self.fc2    = nn.Linear(256, num_classes)

        state_dict = self.load_model('./models/LightCNN_29Layers.pth.tar')
        self.load_state_dict(state_dict)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        # RGB 2 GRAY
        if self.options.channels == 3:
            rgb2gray = transforms.Grayscale()
            x = rgb2gray(x)
        # Denormalize input to range [0;1]
        x = denormalize(x, mean=self.options.normalize[0], std=self.options.normalize[1])
        # Resize to 128x128
        x = F.interpolate(x, size=128)

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)
        p = x

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        out = [p, fc]
        return out


    def load_model(self, path):
        sd = torch.load(path)
        sd = sd['state_dict']
        state_dict = {}
        for k, v in sd.items():
            state_dict[k.replace('module.','')] = v
        del sd
        return state_dict


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out
