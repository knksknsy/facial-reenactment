import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import DataParallel
import numpy as np
from typing import Tuple
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer

from configs.options import Options

def init_weights(m, init_type='normal', gain=0.02):
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


def lr_linear_schedule(current_epoch, epoch_start, epoch_end, lr_base, lr_end):
    mode = -1.0 if lr_base > lr_end else 1.0

    cur_step = (current_epoch - epoch_start)
    if current_epoch >= epoch_start and current_epoch < epoch_end:
        lr_decay = (abs(lr_base - lr_end) / (epoch_end - epoch_start)) * mode
        return lr_base + (lr_decay * cur_step)
    elif current_epoch < epoch_start:
        return lr_base
    elif current_epoch >= epoch_end:
        return lr_end


def init_seed_state(options: Options, model_name: str = 'Generator'):
    # Set seeds
    if options.continue_id is None:
        torch.manual_seed(options.seed)
        np.random.seed(options.seed)
    # Load seed states from checkpoint
    else:
        filename = f'{model_name}_{options.continue_id}'
        state_dict = torch.load(os.path.join(options.checkpoint_dir, filename), map_location=torch.device('cpu') if options.device == 'cpu' else None)
        numpy_seed_state = state_dict['numpy_seed_state'] if 'numpy_seed_state' in state_dict else None
        torch_seed_state = state_dict['torch_seed_state'] if 'torch_seed_state' in state_dict else None
        del state_dict

        if torch_seed_state is not None:
            torch.set_rng_state(torch_seed_state)
        if numpy_seed_state is not None:
            np.random.set_state(numpy_seed_state)


def load_model(model: Module, optimizer: Optimizer, scheduler: object, options: Options) -> Tuple[Module, Optimizer, object, str, str]:
        filename = f'{type(model).__name__}_{options.continue_id}'
        state_dict = torch.load(os.path.join(options.checkpoint_dir, filename), map_location=torch.device('cpu') if options.device == 'cpu' else None)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        if 'scheduler' in state_dict and state_dict['scheduler'] is not None and scheduler is not None:
            scheduler.load_state_dict(state_dict['scheduler'])
        epoch = state_dict['epoch'] + 1
        iteration = state_dict['iteration']

        if options.overwrite_optim:
            optimizer.param_groups[0]['lr'] = options.lr_g if type(model).__name__ == 'Generator' else options.lr_d
            optimizer.param_groups[0]['betas'] = (options.beta1, options.beta2)
            optimizer.param_groups[0]['weight_decay'] = options.weight_decay

        return model, optimizer, scheduler, epoch, iteration


def save_model(model: Module, optimizer: Optimizer, scheduler: object, epoch: str, iteration: str, options: Options, ext='.pth', time_for_name=None):
    if time_for_name is None:
        time_for_name = datetime.now()

    m = model.module if isinstance(model, DataParallel) else model
    # o = optimizer.module if isinstance(optimizer, DataParallel) else optimizer

    m.eval()
    if options.device == 'cuda':
        m.cpu()

    filename = f'{type(m).__name__}_t{time_for_name:%Y%m%d_%H%M}_e{str(epoch).zfill(3)}_i{str(iteration).zfill(8)}{ext}'
    torch.save({
            'model': m.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch,
            'iteration': iteration,
            'numpy_seed_state': np.random.get_state(),
            'torch_seed_state': torch.random.get_rng_state()
        },
        os.path.join(options.checkpoint_dir, filename)
    )

    if options.device == 'cuda':
        m.to(options.device)
    m.train()
