import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn import functional as F
from torchvision.models.vgg import vgg16

from configs import Options
from models.utils import init_weights
from models.vgg import VGG

# TODO: Archtitecture
class Generator(nn.Module):
    def __init__(self, options: Options):
        super(Generator, self).__init__()
        self.options = options

        init_weights(self)
        self.to(self.options.device)


    def forward(self, images, landmarks):
        return images


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


    def loss_adv(self, pred_AB):
        return -pred_AB * self.w_adv


    def loss_rec(self, AB, image2):
        return F.l1_loss(AB, image2)


    def loss_self(self, ABA, image1):
        return F.l1_loss(ABA, image1)


    def loss_triple(self, AC, BC):
        return torch.mean(torch.abs(AC - BC))


    def loss_percep(self, AB, image2):
        vgg_fake = self.VGG(AB)
        vgg_target = self.VGG(image2)
        l_percep = 0

        for idx in range(len(self.VGG.layer_name_mapping)):
            l_percep += F.mse_loss(vgg_fake[idx], vgg_target[idx].detach())
        
        return l_percep


    def loss_tv(self, AB):
        batch_size = AB.size()[0]
        h_AB = AB.size()[2]
        w_AB = AB.size()[3]
        count_h = torch.numel(AB[:,:,1:,:])
        count_w = torch.numel(AB[:,:,:,1:])
        h_tv = torch.pow((AB[:,:,1:,:] - AB[:,:, :h_AB -1,:]),2).sum()
        w_tv = torch.pow((AB[:,:,:,1:] - AB[:,:,:, :w_AB -1]),2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


    def forward(self, image1, image2, AB, pred_AB, ABA, AC, BC):
        image1 = image1.to(self.options.device)
        image2 = image2.to(self.options.device)
        AB = AB.to(self.options.device)
        pred_AB = pred_AB.to(self.options.device)
        ABA = ABA.to(self.options.device)
        AC = AC.to(self.options.device)
        BC = BC.to(self.options.device)

        l_adv = self.loss_adv(pred_AB) * self.w_adv
        l_rec = self.w_rec * self.loss_rec(AB, image2)
        l_self = self.w_self * self.loss_self(ABA, image1)
        l_triple = self.w_triple * self.loss_triple(AC, BC)
        l_percep = self.w_percep * self.loss_percep(AB, image2)
        l_tv = self.w_tv * self.loss_tv(AB)

        return l_adv + l_rec + l_self + l_triple + l_percep + l_tv
