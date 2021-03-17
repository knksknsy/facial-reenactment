from models.lightcnn import LightCNN
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn import functional as F
from torchsummary import summary

import sys
from io import StringIO

from configs import Options
from models.utils import init_weights
from models.vgg import VGG16
from models.components import ConvBlock, DownSamplingBlock, UpSamplingBlock, ResidualBlock
from loggings.logger import Logger

class Generator(nn.Module):
    def __init__(self, options: Options):
        super(Generator, self).__init__()
        self.options = options
        c = self.options.channels
        spec_norm = self.options.spec_norm

        layers = []
        layers.append(ConvBlock(in_channels=c+c, out_channels=64, kernel_size=7, stride=1, padding=3, instance_norm=True, activation='leakyrelu', bias=False, sn=spec_norm))  # B x 64 x 128 x 128
        layers.append(DownSamplingBlock(64,  128, kernel_size=4, stride=2, padding=1, instance_norm=True, activation='leakyrelu', bias=False, sn=spec_norm))                  # B x 128 x 64 x 64
        layers.append(DownSamplingBlock(128, 256, kernel_size=4, stride=2, padding=1, instance_norm=True, activation='leakyrelu', bias=False, sn=spec_norm))                  # B x 256 x 32 x 32

        layers.append(ResidualBlock(256, 256, kernel_size=3, stride=1, padding=1, instance_norm=True, activation='relu', bias=False, sn=spec_norm))                           # B x 256 x 32 x 32
        layers.append(ResidualBlock(256, 256, kernel_size=3, stride=1, padding=1, instance_norm=True, activation='relu', bias=False, sn=spec_norm))                           # B x 256 x 32 x 32
        layers.append(ResidualBlock(256, 256, kernel_size=3, stride=1, padding=1, instance_norm=True, activation='relu', bias=False, sn=spec_norm))                           # B x 256 x 32 x 32
        layers.append(ResidualBlock(256, 256, kernel_size=3, stride=1, padding=1, instance_norm=True, activation='relu', bias=False, sn=spec_norm))                           # B x 256 x 32 x 32
        layers.append(ResidualBlock(256, 256, kernel_size=3, stride=1, padding=1, instance_norm=True, activation='relu', bias=False, sn=spec_norm))                           # B x 256 x 32 x 32
        layers.append(ResidualBlock(256, 256, kernel_size=3, stride=1, padding=1, instance_norm=True, activation='relu', bias=False, sn=spec_norm))                           # B x 256 x 32 x 32

        layers.append(UpSamplingBlock(256, 128, kernel_size=4, stride=2, padding=1, instance_norm=True, activation='leakyrelu', bias=False, sn=spec_norm))                    # B x 128 x 64 x 64
        layers.append(UpSamplingBlock(128,  64, kernel_size=4, stride=2, padding=1, instance_norm=True, activation='leakyrelu', bias=False, sn=spec_norm))                    # B x 64 x 128 x 128

        layers.append(ConvBlock(in_channels=64, out_channels=c, kernel_size=7, stride=1, padding=3, instance_norm=False, activation='tanh', bias=False, sn=spec_norm))        # B x C x 128 x 128
        self.layers = nn.Sequential(*layers)

        self.apply(init_weights)
        self.to(self.options.device)


    def forward(self, images, landmarks):
        # Input: B x C*2 x 128 x 128
        return self.layers(torch.cat((images, landmarks), dim=1))


    def __str__(self):
        old_stdout = sys.stdout
        sys.stdout = new_stdout = StringIO()
        summary(self.layers, input_size=(self.options.channels*2, self.options.image_size, self.options.image_size), batch_size=self.options.batch_size, device=self.options.device)
        sys.stdout = old_stdout
        return new_stdout.getvalue()


class LossG(nn.Module):
    def __init__(self, logger: Logger, options: Options, vgg_device: str, lightcnn_device: str):
        super(LossG, self).__init__()
        self.logger = logger
        self.options = options
        self.vgg_device = vgg_device
        self.lightcnn_device = lightcnn_device

        self.w_adv = self.options.l_adv
        self.w_rec = self.options.l_rec
        self.w_self = self.options.l_self
        self.w_triple = self.options.l_triple
        self.w_id = self.options.l_id
        self.w_percep = self.options.l_percep
        self.w_feature_matching = self.options.l_feature_matching
        self.w_tv = self.options.l_tv

        self.to(self.options.device)

        if self.options.l_percep > 0:
            self.vgg16 = VGG16(self.options, vgg_type=self.options.vgg_type)
            if self.vgg_device == 'cuda' and torch.cuda.device_count() > 1:
                self.vgg16 = DataParallel(self.vgg16)
            self.vgg16.to(self.vgg_device)
            self.vgg16.eval()

        if self.options.l_id > 0:
            self.lightcnn = LightCNN(self.options)
            if self.lightcnn_device == 'cuda' and torch.cuda.device_count() > 1:
                self.lightcnn = DataParallel(self.lightcnn)
            self.lightcnn.to(self.lightcnn_device)
            self.lightcnn.eval()


    def loss_adv(self, d_fake_12):
        return -torch.mean(d_fake_12)


    def loss_rec(self, fake_12, real_2):
        return F.l1_loss(fake_12, real_2)


    def loss_self(self, fake_121, real_1):
        return F.l1_loss(fake_121, real_1)


    def loss_triple(self, fake_13, fake_23):
        return torch.mean(torch.abs(fake_13 - fake_23))


    def loss_id(self, fake_12, real_2):
        lcnn_fake = self.lightcnn(fake_12.to(self.lightcnn_device))
        lcnn_real = self.lightcnn(real_2.to(self.lightcnn_device))
        l_id = 0

        for idx in range(self.lightcnn.output_count):
            l_id += F.l1_loss(lcnn_fake[idx], lcnn_real[idx].detach())
        return l_id


    def loss_percep(self, fake_12, real_2):
        vgg_fake = self.vgg16(fake_12.to(self.vgg_device))
        vgg_real = self.vgg16(real_2.to(self.vgg_device))
        l_percep = 0

        for idx in range(self.vgg16.relu_count):
            l_percep += F.mse_loss(vgg_fake[idx], vgg_real[idx].detach())
        return l_percep


    def loss_feature_matching(self, fm_real_2, fm_fake_12):
        l_feature_matching = 0
        for real, fake in zip(fm_real_2, fm_fake_12):
            l_feature_matching += self.w_feature_matching * torch.abs(real - fake).mean()
        return l_feature_matching


    def loss_tv(self, fake_12):
        batch_size = fake_12.size()[0]
        h_AB = fake_12.size()[2]
        w_AB = fake_12.size()[3]
        count_h = torch.numel(fake_12[:,:,1:,:])
        count_w = torch.numel(fake_12[:,:,:,1:])
        h_tv = torch.pow((fake_12[:,:,1:,:] - fake_12[:,:, :h_AB -1,:]),2).sum()
        w_tv = torch.pow((fake_12[:,:,:,1:] - fake_12[:,:,:, :w_AB -1]),2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


    def forward(self, real_1, real_2, d_fake_12, fake_12, fake_121, fake_13, fake_23, fm_fake_12, fm_real_2):
        l_adv = self.w_adv * self.loss_adv(d_fake_12) if self.w_adv > 0 else 0
        l_rec = self.w_rec * self.loss_rec(fake_12, real_2) if self.w_rec > 0 else 0
        l_self = self.w_self * self.loss_self(fake_121, real_1) if self.w_self > 0 else 0
        l_triple = self.w_triple * self.loss_triple(fake_13, fake_23) if self.w_triple > 0 else 0
        l_id = self.w_id * self.loss_id(fake_12, real_2) if self.w_id > 0 else 0
        l_percep = self.w_percep * self.loss_percep(fake_12, real_2) if self.w_percep > 0 else 0
        l_feature_matching = self.loss_feature_matching(fm_real_2, fm_fake_12) if self.w_feature_matching > 0 else 0
        l_tv = self.w_tv * self.loss_tv(fake_12) if self.w_tv > 0 else 0

        loss_G = l_adv + l_rec + l_self + l_triple + l_id + l_percep + l_feature_matching + l_tv

        # LOSSES DICT
        losses_G = dict({
            'Loss_G': loss_G.detach().item(),
            'Loss_Adv': l_adv.detach().item() if self.w_adv > 0 else None,
            'Loss_Rec': l_rec.detach().item() if self.w_rec > 0 else None,
            'Loss_Self': l_self.detach().item() if self.w_self > 0 else None,
            'Loss_Triple': l_triple.detach().item() if self.w_triple > 0 else None,
            'Loss_Id': l_id.detach().item() if self.w_id > 0 else None,
            'Loss_Percep': l_percep.detach().item() if self.w_percep > 0 else None,
            'Loss_Feature_Matching': l_feature_matching.detach().item() if self.w_feature_matching > 0 else None,
            'Loss_TV': l_tv.detach().item() if self.w_tv > 0 else None
        })
        del l_adv, l_rec, l_self, l_triple, l_id, l_percep, l_feature_matching, l_tv

        return loss_G, losses_G
