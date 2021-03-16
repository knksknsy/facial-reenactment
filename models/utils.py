import os
import torch
import torch.nn as nn

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


def load_seed_state(options: Options, model_name: str = 'Generator'):
        filename = f'{model_name}_{options.continue_id}'
        state_dict = torch.load(os.path.join(options.checkpoint_dir, filename), map_location=torch.device('cpu') if options.device == 'cpu' else None)
        numpy_seed_state = state_dict['numpy_seed_state'] if 'numpy_seed_state' in state_dict else None
        torch_seed_state = state_dict['torch_seed_state'] if 'torch_seed_state' in state_dict else None
        del state_dict
        return numpy_seed_state, torch_seed_state
