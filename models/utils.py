import os
import logging
import torch
import torch.nn as nn
import cv2

from torch.nn import DataParallel
from datetime import datetime

from configs import Options

def init_weights(model, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'initialization method {init_type} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def save_image(path, filename, data, epoch=None, batch_num=None):
    if not os.path.isdir(path):
        os.makedirs(path)

    data = data.clone().detach().cpu()
    img = (data.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype('uint8')

    if epoch is not None and batch_num is not None:
        filename = f'e_{epoch}_b{batch_num}_{filename}'
    
    cv2.imwrite(os.path.join(path, filename), img)


def imshow(data):
    data = data.clone().detach().cpu()
    img = (data.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype('uint8')
    cv2.imshow('Image Preview', img)
