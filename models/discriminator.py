import torch
import torch.nn as nn
from torchsummary import summary

import sys
from io import StringIO

from configs import Options
from models.utils import init_weights
from models.components import ConvBlock
from loggings.logger import Logger

class Discriminator(nn.Module):
    def __init__(self, options: Options):
        super(Discriminator, self).__init__()
        self.options = options

        layers = []
        layers.append(ConvBlock(3,      64, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu'))           # B x   64 x 64 x 64
        layers.append(ConvBlock(64,    128, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu'))           # B x  128 x 32 x 32
        layers.append(ConvBlock(128,   256, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu'))           # B x  256 x 16 x 16
        layers.append(ConvBlock(256,   512, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu'))           # B x  512 x  8 x  8
        layers.append(ConvBlock(512,  1024, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu'))           # B x 1024 x  4 x  4
        layers.append(ConvBlock(1024, 2048, kernel_size=4, stride=2, padding=1, instance_norm=False, activation='leakyrelu'))           # B x 2048 x  2 x  2
        layers.append(ConvBlock(2048,    1, kernel_size=3, stride=1, padding=1, instance_norm=False, activation=None, use_bias=False))  # B x    1 x  2 x  2
        self.layers = nn.Sequential(*layers)

        self.apply(init_weights)
        self.to(self.options.device)


    def forward(self, x):
        # Input:    B x 3 x 128 x 128
        # Output:   B x 2 x 2
        return self.layers(x).squeeze()


    def __str__(self):
        old_stdout = sys.stdout
        sys.stdout = new_stdout = StringIO()
        summary(self.layers, input_size=(3, 128, 128), batch_size=self.options.batch_size, device=self.options.device)
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


    # TODO: check gradient penalty implementation
    def loss_gp(self, discriminator, real, fake, req_grad):
        alpha = torch.rand(real.size(0), 1, 1, 1).to(self.options.device).expand_as(real)
        interpolated = alpha * real + (1 - alpha) * fake
        if req_grad: interpolated.requires_grad = True 
        prob_interpolated = discriminator(interpolated)

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
        return l_gp


    def forward(self, discriminator, d_fake, d_real, fake, real, req_grad=False):
        l_adv_real = self.loss_adv_real(d_real)
        l_adv_fake = self.loss_adv_fake(d_fake)
        l_gp = self.w_gp * self.loss_gp(discriminator, real, fake, req_grad)
        
        loss_D = l_adv_real + l_adv_fake + l_gp

        # LOSSES DICT
        losses_D = dict({
            'Loss_D': loss_D.detach().item(),
            'Loss_Adv_Real': l_adv_real.detach().item(),
            'Loss_Adv_Fake': l_adv_fake.detach().item(),
            'Loss_GP': l_gp.detach().item()
        })
        del l_adv_real, l_adv_fake, l_gp

        return loss_D, losses_D
