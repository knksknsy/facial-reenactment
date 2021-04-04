from typing import Tuple

import torch
from torch.nn import DataParallel
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer

from configs import Options
from loggings.logger import Logger
from models.utils import load_model, save_model

class NetworkDetection(nn.Module):
    def __init__(self, logger: Logger, options: Options, model_path=None):
        super(NetworkDetection, self).__init__()
        self.logger = logger
        self.model_path = model_path
        self.options = options

        self.continue_epoch = 0
        self.continue_iteration = 0

        # Testing mode
        if self.model_path is not None:
            state_dict = torch.load(self.model_path)
            self.load_state_dict(state_dict['model'])
            self.continue_epoch = state_dict['epoch']

            self.criterion = LossDetection(self.logger, self.options)

        # Training mode
        else:
            # Print model summaries
            self.logger.log_info('===== DETECTOR ARCHITECTURE =====')
            self.logger.log_info(self)

            # Load networks into multiple GPUs
            if torch.cuda.device_count() > 1:
                self = DataParallel(self)

            self.criterion = LossDetection(self.logger, self.options)

            self.optimizer = Adam(
                params=self.parameters(),
                lr=self.options.lr,
                betas=(self.options.beta1, self.options.beta2),
                weight_decay=self.options.weight_decay
            )

            # Step based LR decay schedule
            if 'lr_step_decay' in self.options.config['train']['optimizer']:
                self.scheduler = StepLR(
                    optimizer=self.optimizer,
                    step_size=self.options.step_size,
                    gamma=self.options.gamma
                )

            # Plateau based LR decay schedule
            elif 'lr_plateau_decay' in self.options.config['train']['optimizer']:
                self.scheduler = ReduceLROnPlateau(
                    optimizer=self.optimizer,
                    mode=self.options.plateau_mode,
                    factor=self.options.plateau_factor,
                    patience=self.options.plateau_patience,
                    cooldown=2*self.options.plateau_patience,
                    min_lr=self.options.plateau_min_lr
                )

            if self.options.continue_id is not None:
                self.optimizer, self.scheduler, self.continue_epoch, self.continue_iteration = self.load_model(self.optimizer, self.scheduler, self.options)


    def __call__(self, images_real, images_fake, labels_real, labels_fake, batch = None, calc_loss: bool = True):
        # During inference
        if not calc_loss:
            with torch.no_grad():
                return self.forward(batch)
        # During testing
        else:
            preds, features, loss, losses_dict = self.get_loss(images_real, images_fake, labels_real, labels_fake)
            return preds, features, loss, losses_dict


    def forward(self, batch):
        self.zero_grad()

        # Forward

        loss, losses_dict = self.criterion()
        loss.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.options.grad_clip, norm_type=2)

        self.optimizer.step()

        return None


    def get_loss(self, batch):
        with torch.no_grad():
            # Forward

            loss, losses_dict = self.criterion_G()
            return None


    def load_model(self, model: Module, optimizer: Optimizer, scheduler: object, options: Options) -> Tuple[Module, Optimizer, object, str, str]:
        filename = f'{type(model).__name__}_{options.continue_id}'
        load_model(model, optimizer, scheduler, options)
        self.logger.log_info(f'Model loaded: {filename}')


    def save_model(self, model: Module, optimizer: Optimizer, scheduler: object, epoch: str, iteration: str, options: Options, ext='.pth', time_for_name=None):
        filename = f'{type(model).__name__}_t{time_for_name:%Y%m%d_%H%M}_e{str(epoch).zfill(3)}_i{str(iteration).zfill(8)}{ext}'
        save_model(model, optimizer, scheduler, epoch, iteration, options, ext, time_for_name)
        self.logger.log_info(f'Model saved: {filename}')


class LossDetection(nn.Module):
    def __init__(self, logger: Logger, options: Options):
        super(LossDetection, self).__init__()
        self.logger = logger
        self.options = options

        self.to(self.options.device)


    def forward():
        pass
