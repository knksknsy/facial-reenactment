import torch
import torch.nn as nn
from torch.nn import functional as F

from configs import Options
from models.utils import init_weights

# TODO: Archtitecture
class Discriminator(nn.Module):
    def __init__(self, options: Options):
        super(Discriminator, self).__init__()
        self.options = options
        
        init_weights(self)
        self.to(self.options.device)


    def forward(self):
        return 0


class LossD(nn.Module):
    def __init__(self, options: Options):
        super(LossD, self).__init__()
        self.options = options
        self.to(self.options.device)


    def loss_adv_real(self, real_AB):
        return -torch.mean(real_AB)


    def loss_adv_fake(self, fake_AB):
        return torch.mean(fake_AB)


    def loss_gp(self, D, real_AB, fake_AB):
        alpha = torch.rand(real_AB.size(0),1,1,1).to(self.options.device).expand_as(real_AB)
        interpolated = alpha * real_AB + (1 - alpha) * fake_AB
        interpolated.requires_grad = True
        prob_interpolated = D(interpolated)

        grad = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]

        grad = grad.view(grad.size(0),-1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        l_gp = torch.mean((grad_l2norm-1)**2)
        return l_gp


    def forward(self, D, fake_AB, real_AB, AB, image2):
        fake_AB = fake_AB.to(self.options.device)
        real_AB = real_AB.to(self.options.device)
        AB = AB.to(self.options.device)
        image2 = image2.to(self.options.device)

        l_adv_real = self.loss_adv_real(real_AB)
        l_adv_fake = self.loss_adv_fake(fake_AB)
        l_gp = 10 * self.loss_gp(image2, AB)

        return l_adv_real + l_adv_fake + l_gp
