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
from loggings.logger import Logger

# TODO: test leakyrelu
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
        self.layers = nn.Sequential(*layers)

        color_layers = []
        color_layers.append(ConvBlock(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3, instance_norm=False, activation='tanh', use_bias=False))   # B x 3 x 128 x 128
        self.color_layers = nn.Sequential(*color_layers)

        mask_layers = []
        mask_layers.append(ConvBlock(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, instance_norm=False, activation='sigmoid', use_bias=False)) # B x 1 x 128 x 128
        self.mask_layers = nn.Sequential(*mask_layers)

        self.apply(init_weights)
        self.to(self.options.device)


    def forward(self, images, landmarks):
        # Input: B x 6 x 128 x 128
        features = self.layers(torch.cat((images, landmarks), dim=1))
        color = self.color_layers(features)
        mask = self.mask_layers(features)

        output = mask * images + (1 - mask) * color
        return output, mask, color


    def __str__(self):
        old_stdout = sys.stdout
        sys.stdout = new_stdout = StringIO()
        summary(self.layers, input_size=(6, 128, 128), batch_size=self.options.batch_size, device=self.options.device)
        sys.stdout = old_stdout
        return new_stdout.getvalue()


class LossG(nn.Module):
    def __init__(self, logger: Logger, options: Options, vgg_device='cpu'):
        super(LossG, self).__init__()
        self.logger = logger
        self.options = options
        self.vgg_device = vgg_device

        self.w_adv = self.options.l_adv
        self.w_rec = self.options.l_rec
        self.w_self = self.options.l_self
        self.w_triple = self.options.l_triple
        self.w_percep = self.options.l_percep
        self.w_tv = self.options.l_tv
        self.w_mask = self.options.l_mask
        self.w_mask_smooth = self.options.l_mask_smooth

        self.to(self.options.device)

        self.VGG = VGG(vgg16(pretrained=True))
        if self.vgg_device == 'cuda' and torch.cuda.device_count() > 1:
            self.VGG = DataParallel(self.VGG)
        self.VGG.to(self.vgg_device)
        self.VGG.eval()


    def loss_adv(self, d_fake_12):
        return -torch.mean(d_fake_12)


    def loss_rec(self, fake_12, real_2):
        return F.l1_loss(fake_12, real_2)


    def loss_self(self, fake_121, real_1):
        return F.l1_loss(fake_121, real_1)


    def loss_triple(self, fake_13, fake_23):
        return torch.mean(torch.abs(fake_13 - fake_23))


    def loss_percep(self, fake_12, real_2):
        vgg_fake = self.VGG(fake_12.to(self.vgg_device))
        vgg_real = self.VGG(real_2.to(self.vgg_device))
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


    def loss_mask(self, fake_mask_12, fake_mask_121, fake_mask_13, fake_mask_23):
        l_m_12 = self._loss_mask(fake_mask_12)
        l_m_121 = self._loss_mask(fake_mask_121)
        l_m_13 = self._loss_mask(fake_mask_13)
        l_m_23 = self._loss_mask(fake_mask_23)
        l_ms_12 = self._loss_mask_smooth(fake_mask_12)
        l_ms_121 = self._loss_mask_smooth(fake_mask_121)
        l_ms_13 = self._loss_mask_smooth(fake_mask_13)
        l_ms_23 = self._loss_mask_smooth(fake_mask_23)

        return l_m_12 + l_m_121 + l_m_13 + l_m_23 + l_ms_12 + l_ms_121 + l_ms_13 + l_ms_23


    def _loss_mask(self, mask):
        return torch.mean(mask) * self.w_mask


    def _loss_mask_smooth(self, mask):
        return (
            torch.sum(torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:])) + torch.sum(torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :]))
        ) * self.w_mask_smooth


    def forward(self, real_1, real_2, d_fake_12, fake_12, fake_121, fake_13, fake_23, fake_mask_12, fake_mask_121, fake_mask_13, fake_mask_23, iterations: int):
        l_adv = self.w_adv * self.loss_adv(d_fake_12)
        l_rec = self.w_rec * self.loss_rec(fake_12, real_2)
        l_self = self.w_self * self.loss_self(fake_121, real_1)
        l_triple = self.w_triple * self.loss_triple(fake_13, fake_23)
        l_percep = self.w_percep * self.loss_percep(fake_12, real_2)
        l_tv = self.w_tv * self.loss_tv(fake_12)
        l_mask = self.loss_mask(fake_mask_12, fake_mask_121, fake_mask_13, fake_mask_23)

        loss_G = l_adv + l_rec + l_self + l_triple + l_percep + l_tv + l_mask

        # LOG LOSSES
        losses_G = dict({
            'Loss_G': loss_G.detach().item(),
            'Loss_Adv': l_adv.detach().item(),
            'Loss_Rec': l_rec.detach().item(),
            'Loss_Self': l_self.detach().item(),
            'Loss_Triple': l_triple.detach().item(),
            'Loss_Percep': l_percep.detach().item(),
            'Loss_TV': l_tv.detach().item(),
            'Loss_Mask': l_mask.detach().item()
        })
        self.logger.log_scalars(losses_G, iterations)
        del losses_G, l_adv, l_rec, l_self, l_triple, l_percep, l_tv, l_mask

        return loss_G
