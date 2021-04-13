import torch
import torch.nn as nn
from torchsummary import summary

import sys
from io import StringIO

from configs.options import Options
from utils.models import init_weights
from loggings.logger import Logger
from ..models.components import ConvBlock

class Discriminator(nn.Module):
    def __init__(self, options: Options):
        super(Discriminator, self).__init__()
        self.options = options
        c = self.options.channels
        spec_norm = self.options.spec_norm

        down_blocks = []
        if self.options.conv_blocks_d == 6:
            down_blocks.append(ConvBlock(c,      64, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu', sn=spec_norm))  # B x   64 x 64 x 64
            down_blocks.append(ConvBlock(64,    128, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu', sn=spec_norm))  # B x  128 x 32 x 32
            down_blocks.append(ConvBlock(128,   256, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu', sn=spec_norm))  # B x  256 x 16 x 16
            down_blocks.append(ConvBlock(256,   512, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu', sn=spec_norm))  # B x  512 x  8 x  8
            down_blocks.append(ConvBlock(512,  1024, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu', sn=spec_norm))  # B x 1024 x  4 x  4
            down_blocks.append(ConvBlock(1024, 2048, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu', sn=spec_norm))  # B x 2048 x  2 x  2
        elif self.options.conv_blocks_d == 4:
            down_blocks.append(ConvBlock(c,    64, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu', sn=spec_norm))    # B x   64 x 64 x 64
            down_blocks.append(ConvBlock(64,  128, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu', sn=spec_norm))    # B x  128 x 32 x 32
            down_blocks.append(ConvBlock(128, 256, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu', sn=spec_norm))    # B x  256 x 16 x 16
            down_blocks.append(ConvBlock(256, 512, kernel_size=4, stride=1, padding=1, instance_norm=False, activation='leakyrelu', sn=spec_norm))    # B x  512 x 15 x 15
        else:
            raise NotImplementedError()
        self.down_blocks = nn.Sequential(*down_blocks)
        
        conv = []
        if self.options.conv_blocks_d == 6:
            conv.append(ConvBlock(2048,    1, kernel_size=3, stride=1, padding=1, instance_norm=False, activation=None, bias=False))                  # B x 1 x   2 x  2
        elif self.options.conv_blocks_d == 4:
            conv.append(ConvBlock(512, 1, kernel_size=4, stride=1, padding=1, instance_norm=False, activation=None, bias=False, sn=spec_norm))        # B x 1 x  14 x  14
        self.conv = nn.Sequential(*conv)

        self.apply(init_weights)
        self.to(self.options.device)


    def forward(self, x):
        # Input:    B x C x 128 x 128
        feature_maps = []
        out = x
        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        # Output:   B x 14 x 14
        prediction_map = self.conv(out).squeeze()
        return feature_maps, prediction_map


    def __str__(self):
        old_stdout = sys.stdout
        sys.stdout = new_stdout = StringIO()
        summary(self, input_size=(self.options.channels, self.options.image_size, self.options.image_size), batch_size=self.options.batch_size, device=self.options.device)
        sys.stdout = old_stdout
        return new_stdout.getvalue()


class LossD(nn.Module):
    def __init__(self, logger: Logger, options: Options):
        super(LossD, self).__init__()
        self.logger = logger
        self.options = options
        self.w_gp = self.options.l_gp

        self.to(self.options.device)


    def loss_adv_real(self, d_real):
        return -torch.mean(d_real)


    def loss_adv_fake(self, d_fake):
        return torch.mean(d_fake)


    def loss_gp(self, discriminator, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, 1).to(self.options.device).expand_as(real)
        interpolated = alpha * real + (1 - alpha) * fake
        # if req_grad: interpolated.requires_grad = True 
        fm_prob_interpolated, prob_interpolated = discriminator(interpolated)

        grad = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.options.device),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        l_gp = torch.mean((grad_l2norm - 1) ** 2)
        del alpha, interpolated, fm_prob_interpolated, prob_interpolated, grad, grad_l2norm
        return l_gp


    def forward(self, discriminator, d_fake, d_real, fake, real, skip_gp=False):
        l_adv_real = self.loss_adv_real(d_real)
        l_adv_fake = self.loss_adv_fake(d_fake)
        l_gp = self.w_gp * self.loss_gp(discriminator, real, fake) if self.w_gp > 0 and not skip_gp else 0
        
        loss_D = l_adv_real + l_adv_fake + l_gp

        # LOSSES DICT
        losses_D = dict({
            'Loss_D': loss_D.detach().item(),
            'Loss_Adv_Real': l_adv_real.detach().item(),
            'Loss_Adv_Fake': l_adv_fake.detach().item(),
            'Loss_GP': l_gp.detach().item() if self.w_gp > 0 and not skip_gp else None
        })
        del l_adv_real, l_adv_fake, l_gp

        return loss_D, losses_D
