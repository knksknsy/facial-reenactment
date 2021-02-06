import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn import functional as F
from torchvision.models.vgg import vgg16
from torchsummary import summary

import sys
from io import StringIO

from configs import Options
from models.utils import init_weights
from models.vgg import VGG
from models.components import ConvBlock, DownSamplingBlock, UpSamplingBlock, ResidualBlock

class Generator(nn.Module):
    def __init__(self, options: Options):
        super(Generator, self).__init__()
        self.options = options

        layers = []
        layers.append(ConvBlock(in_channels=3+3, out_channels=64, kernel_size=7, stride=1, padding=3)) # B x 64 x 128 x 128

        layers.append(DownSamplingBlock(64, 128))   # B x 128 x 64 x 64
        layers.append(DownSamplingBlock(128, 256))  # B x 256 x 32 x 32

        layers.append(ResidualBlock(256, 256))      # B x 256 x 32 x 32
        layers.append(ResidualBlock(256, 256))      # B x 256 x 32 x 32
        layers.append(ResidualBlock(256, 256))      # B x 256 x 32 x 32
        layers.append(ResidualBlock(256, 256))      # B x 256 x 32 x 32
        layers.append(ResidualBlock(256, 256))      # B x 256 x 32 x 32
        layers.append(ResidualBlock(256, 256))      # B x 256 x 32 x 32

        layers.append(UpSamplingBlock(256, 128))    # B x 128 x 64 x 64
        layers.append(UpSamplingBlock(128, 64))     # B x 64 x 128 x 128

        layers.append(ConvBlock(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3, instance_norm=False, activation='tanh')) # B x 3 x 128 x 128

        self.layers = nn.Sequential(*layers)

        self.apply(init_weights)
        self.to(self.options.device)


    def forward(self, images, landmarks):
        # Input: B x 6 x 128 x 128
        return self.layers(torch.cat((images, landmarks), dim=1))


    def __str__(self):
        old_stdout = sys.stdout
        sys.stdout = new_stdout = StringIO()
        summary(self.layers, input_size=(6, 128, 128), batch_size=self.options.batch_size, device=self.options.device)
        sys.stdout = old_stdout
        return new_stdout.getvalue()


class LossG(nn.Module):
    def __init__(self, options: Options):
        super(LossG, self).__init__()
        self.options = options

        self.w_adv = self.options.l_adv
        self.w_rec = self.options.l_rec
        self.w_self = self.options.l_self
        self.w_triple = self.options.l_triple
        self.w_percep = self.options.l_percep
        self.w_tv = self.options.l_tv

        self.to(self.options.device)

        self.VGG = VGG(vgg16(pretrained=True))
        if torch.cuda.device_count() > 1:
            self.VGG = DataParallel(self.VGG)
            self.VGG.eval()
            self.VGG.to(self.options.device)


    def loss_adv(self, d_fake_12):
        return torch.mean(-d_fake_12) * self.w_adv


    def loss_rec(self, fake_12, real_2):
        return F.l1_loss(fake_12, real_2)


    def loss_self(self, fake_121, real_1):
        return F.l1_loss(fake_121, real_1)


    def loss_triple(self, fake_13, fake_23):
        return torch.mean(torch.abs(fake_13 - fake_23))


    def loss_percep(self, fake_12, real_2):
        vgg_fake = self.VGG(fake_12)
        vgg_real = self.VGG(real_2)
        l_percep = 0

        for idx in range(len(self.VGG.layer_name_mapping)):
            l_percep += F.mse_loss(vgg_fake[idx], vgg_real[idx].detach())
        
        return l_percep


    def loss_tv(self, fake_12):
        batch_size = fake_12.size()[0]
        h_AB = fake_12.size()[2]
        w_AB = fake_12.size()[3]
        count_h = torch.numel(fake_12[:,:,1:,:])
        count_w = torch.numel(fake_12[:,:,:,1:])
        h_tv = torch.pow((fake_12[:,:,1:,:] - fake_12[:,:, :h_AB -1,:]),2).sum()
        w_tv = torch.pow((fake_12[:,:,:,1:] - fake_12[:,:,:, :w_AB -1]),2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


    def forward(self, real_1, real_2, fake_12, d_fake_12, fake_121, fake_13, fake_23):
        real_1 = real_1.to(self.options.device)
        real_2 = real_2.to(self.options.device)
        fake_12 = fake_12.to(self.options.device)
        d_fake_12 = d_fake_12.to(self.options.device)
        fake_121 = fake_121.to(self.options.device)
        fake_13 = fake_13.to(self.options.device)
        fake_23 = fake_23.to(self.options.device)

        l_adv = self.loss_adv(d_fake_12)
        l_rec = self.w_rec * self.loss_rec(fake_12, real_2)
        l_self = self.w_self * self.loss_self(fake_121, real_1)
        l_triple = self.w_triple * self.loss_triple(fake_13, fake_23)
        l_percep = self.w_percep * self.loss_percep(fake_12, real_2)
        l_tv = self.w_tv * self.loss_tv(fake_12)

        return l_adv + l_rec + l_self + l_triple + l_percep + l_tv
