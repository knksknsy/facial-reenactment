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
from models.siamese_resnet import LossSiamese, SiameseResNet
from dataset.dataset import get_pair_classification, get_pair_feature

class NetworkDetection():
    def __init__(self, logger: Logger, options: Options, model_path=None):
        super(NetworkDetection, self).__init__()
        self.logger = logger
        self.model_path = model_path
        self.options = options

        self.continue_epoch = 0
        self.continue_iteration = 0

        # Testing mode
        if self.model_path is not None:
            self.siamese_net = SiameseResNet(self.options, len_feature=self.options.len_feature)
            state_dict = torch.load(self.model_path)
            self.siamese_net.load_state_dict(state_dict['model'])
            self.continue_epoch = state_dict['epoch']

            self.criterion = LossSiamese(self.logger, type=self.options.loss_type, margin=self.options.margin)

        # Training mode
        else:
            self.siamese_net = SiameseResNet(self.options, len_feature=self.options.len_feature)

            # Print model summaries
            self.logger.log_info('===== SIAMESE NETWORK ARCHITECTURE =====')
            self.logger.log_info(self.siamese_net)

            # Load networks into multiple GPUs
            if torch.cuda.device_count() > 1:
                self = DataParallel(self.siamese_net)

            self.criterion = LossSiamese(self.options, type=self.options.loss_type, margin=self.options.margin)

            self.optimizer = Adam(
                params=self.siamese_net.parameters(),
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
                    min_lr=self.options.plateau_min_lr
                )

            if self.options.continue_id is not None:
                self.optimizer, self.scheduler, self.continue_epoch, self.continue_iteration = self.load_model(self.siamese_net, self.optimizer, self.scheduler, self.options)


    def __call__(self, images, labels=None):
        # During inference
        if labels is not None:
            with torch.no_grad():
                return self.siamese_net.forward_classification(images)
        # During testing
        else:
            preds = self.siamese_net.forward_classification(images)
            loss = self.criterion.bce_loss(preds, labels)
            preds, loss


    def forward_feature(self, x1, x2, target):
        self.siamese_net.zero_grad()
        x1, x2 = self.siamese_net.forward_feature(x1, x2)
        loss = self.criterion.loss(x1, x2, target)
        return self.backward(loss)


    def forward_classification(self, x, target):
        self.siamese_net.zero_grad()
        prediction = self.siamese_net.forward_classification(x)
        loss = self.criterion.bce_loss(prediction, target)
        return self.backward(loss), prediction


    def backward(self, loss):
        loss.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.options.grad_clip, norm_type=2)

        self.optimizer.step()
        return loss.detach().item()


    def train(self):
        self.siamese_net.train()


    def eval(self):
        self.siamese_net.eval()


    def load_model(self, model: Module, optimizer: Optimizer, scheduler: object, options: Options) -> Tuple[Module, Optimizer, object, str, str]:
        filename = f'{type(model).__name__}_{options.continue_id}'
        load_model(model, optimizer, scheduler, options)
        self.logger.log_info(f'Model loaded: {filename}')


    def save_model(self, model: Module, optimizer: Optimizer, scheduler: object, epoch: str, iteration: str, options: Options, ext='.pth', time_for_name=None):
        filename = f'{type(model).__name__}_t{time_for_name:%Y%m%d_%H%M}_e{str(epoch).zfill(3)}_i{str(iteration).zfill(8)}{ext}'
        save_model(model, optimizer, scheduler, epoch, iteration, options, ext, time_for_name)
        self.logger.log_info(f'Model saved: {filename}')
