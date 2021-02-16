import os
from typing import Tuple
from datetime import datetime

import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer

from configs import Options
from models.generator import Generator, LossG
from models.discriminator import Discriminator, LossD
from models.utils import lr_linear_decrease
from loggings.logger import Logger

class Network():
    def __init__(self, logger: Logger, options: Options, model_path=None):
        self.logger = logger
        self.model_path = model_path
        self.options = options

        self.continue_epoch = 0
        self.continue_iteration = 0

        # Testing mode
        if self.model_path is not None:
            self.G = Generator(self.options)
            state_dict = torch.load(os.path.join(self.options.checkpoint_dir, self.model_path))
            self.G.load_state_dict(state_dict['model'])

        # Training mode
        else:
            self.G = Generator(self.options)
            self.D = Discriminator(self.options)
            
            # Print model summaries
            self.logger.log_info('===== GENERATOR ARCHITECTURE =====')
            self.logger.log_info(self.G)
            self.logger.log_info('===== DISCRIMINATOR ARCHITECTURE =====')
            self.logger.log_info(self.D)

            # Load networks into multiple GPUs
            if torch.cuda.device_count() > 1:
                self.G = DataParallel(self.G)
                self.D = DataParallel(self.D)

            self.criterion_G = LossG(self.logger, self.options, vgg_device=self.options.device)
            self.criterion_D = LossD(self.logger, self.options)

            self.optimizer_G = Adam(
                params=self.G.parameters(),
                lr=self.options.lr_g,
                betas=(self.options.beta1, self.options.beta1),
                weight_decay=self.options.weight_decay
            )

            self.optimizer_D = Adam(
                params=self.D.parameters(),
                lr=self.options.lr_d,
                betas=(self.options.beta1, self.options.beta1),
                weight_decay=self.options.weight_decay
            )

            lr_lambda_G = lr_linear_decrease(
                epoch_start=self.options.scheduler_epoch_range[0],
                epoch_end=self.options.scheduler_epoch_range[1],
                lr_base=self.options.lr_g,
                lr_min=self.options.scheduler_lr_min
            )
            self.scheduler_G = LambdaLR(
                optimizer=self.optimizer_G,
                lr_lambda=lr_lambda_G
            )

            lr_lambda_D = lr_linear_decrease(
                epoch_start=self.options.scheduler_epoch_range[0],
                epoch_end=self.options.scheduler_epoch_range[1],
                lr_base=self.options.lr_d,
                lr_min=self.options.scheduler_lr_min
            )
            self.scheduler_D = LambdaLR(
                optimizer=self.optimizer_D,
                lr_lambda=lr_lambda_D
            )

            if self.options.continue_id is not None:
                self.G, self.optimizer_G, self.scheduler_G, self.continue_epoch, self.continue_iteration = self.load_model(self.G, self.optimizer_G, self.scheduler_G, self.options)
                self.D, self.optimizer_D, self.scheduler_D, self.continue_epoch, self.continue_iteration = self.load_model(self.D, self.optimizer_D, self.scheduler_D, self.options)


    def __call__(self, images, landmarks):
        with torch.no_grad():
            return self.G(images, landmarks)


    def forward_G(self, batch, iterations: int):
        for p in self.D.parameters():
            p.requires_grad = False

        self.G.zero_grad()

        fake_12, fake_mask_12, _ = self.G(batch['image1'], batch['landmark2'])
        d_fake_12 = self.D(fake_12)
        fake_121, fake_mask_121, _ = self.G(fake_12, batch['landmark1'])
        fake_13, fake_mask_13, _ = self.G(batch['image1'], batch['landmark3'])
        fake_23, fake_mask_23, _ = self.G(fake_12, batch['landmark3'])

        loss_G = self.criterion_G(
            batch['image1'], batch['image2'], d_fake_12,
            fake_12, fake_121, fake_13, fake_23,
            fake_mask_12, fake_mask_121, fake_mask_13, fake_mask_23, iterations
        )
        loss_G.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1, norm_type=2)

        self.optimizer_G.step()

        del d_fake_12, fake_mask_12, fake_121, fake_mask_121, fake_13, fake_mask_13, fake_23, fake_mask_23, _
        
        return fake_12, loss_G


    def forward_D(self, batch, iterations: int):
        for p in self.D.parameters():
            p.requires_grad = True

        self.D.zero_grad()

        fake_12, fake_mask_12, _ = self.G(batch['image1'], batch['landmark2'])
        fake_12 = fake_12.detach()
        fake_12.requires_grad = True
        d_fake_12 = self.D(fake_12)

        d_real_12 = self.D(batch['image2'])

        loss_D, l_adv_real, l_adv_fake = self.criterion_D(self.D, d_fake_12, d_real_12, fake_12, batch['image2'], iterations)
        loss_D.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1, norm_type=2)

        self.optimizer_D.step()

        del fake_12, fake_mask_12, _

        return loss_D, d_real_12, d_fake_12


    def train(self):
        self.G.train()
        self.D.train()


    def eval(self):
        self.G.eval()
        if self.model_path is None:
            self.D.eval()


    def load_model(self, model: Module, optimizer: Optimizer, scheduler: LambdaLR,  options: Options) -> Tuple[Module, Optimizer, LambdaLR, str, str]:
            filename = f'{type(model).__name__}_{options.continue_id}'
            state_dict = torch.load(os.path.join(options.checkpoint_dir, filename))
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['scheduler'])
            epoch = state_dict['epoch'] + 1
            iteration = state_dict['iteration']

            self.logger.log_info(f'Model loaded: {filename}')
            
            return model, optimizer, scheduler, epoch, iteration


    def save_model(self, model: Module, optimizer: Optimizer, scheduler: LambdaLR, epoch: str, iteration: str, options: Options, ext='.pth', time_for_name=None):
        if time_for_name is None:
            time_for_name = datetime.now()

        m = model.module if isinstance(model, DataParallel) else model
        # o = optimizer.module if isinstance(optimizer, DataParallel) else optimizer
        # s = scheduler.module if isinstance(scheduler, DataParallel) else scheduler

        m.eval()
        if options.device == 'cuda':
            m.cpu()

        filename = f'{type(m).__name__}_t{time_for_name:%Y%m%d_%H%M}_e{str(epoch).zfill(3)}_i{str(iteration).zfill(8)}{ext}'
        torch.save({
                'model': m.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'iteration': iteration
            },
            os.path.join(options.checkpoint_dir, filename)
        )

        if options.device == 'cuda':
            m.to(options.device)
        m.train()

        self.logger.log_info(f'Model saved: {filename}')
