import os
from typing import Tuple
from datetime import datetime

import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer

from configs import Options
from models.generator import Generator, LossG
from models.discriminator import Discriminator, LossD
from logger import Logger

class Network():
    def __init__(self, logger: Logger, options: Options, training=False):
        self.logger = logger
        self.training = training
        self.options = options

        self.continue_epoch = 0
        self.continue_iteration = 0

        # Training mode
        if self.training:
            self.G = Generator(self.options)
            self.D = Discriminator(self.options)
            
            # Print model summaries
            self.logger.log_info('Generator architecture:')
            self.logger.log_info(self.G)
            self.logger.log_info('Discriminator architecture:')
            self.logger.log_info(self.D)

            # Load networks into multiple GPUs
            if torch.cuda.device_count() > 1:
                self.G = DataParallel(self.G)
                self.D = DataParallel(self.D)

            self.criterion_G = LossG(self.options)
            self.criterion_D = LossD(self.options)

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

            self.scheduler_G = StepLR(
                optimizer=self.optimizer_G,
                step_size=self.options.scheduler_step_size,
                gamma=self.options.scheduler_gamma
            )

            self.scheduler_D = StepLR(
                optimizer=self.optimizer_D,
                step_size=self.options.scheduler_step_size,
                gamma=self.options.scheduler_gamma
            )

            if self.options.continue_id is not None:
                self.G, self.optimizer_G, self.scheduler_G, self.continue_epoch, self.continue_iteration = self.load_model(self.G, self.optimizer_G, self.scheduler_G, self.options)
                self.D, self.optimizer_D, self.scheduler_D, self.continue_epoch, self.continue_iteration = self.load_model(self.D, self.optimizer_D, self.scheduler_D, self.options)

        # Testing mode
        else:
            self.G = Generator(self.options)
            state_dict = torch.load(self.options.model)
            self.G.load_state_dict(state_dict['model'])


    def __call__(self, images, landmarks):
        with torch.no_grad():
            return self.G(images, landmarks)


    def forward_G(self, batch):
        for p in self.D.parameters():
            p.requires_grad = False

        self.G.zero_grad()

        fake_12 = self.G(batch['image1'], batch['landmark2'])
        d_fake_12 = self.D(fake_12)
        fake_121 = self.G(fake_12, batch['landmark1'])
        fake_13 = self.G(batch['image1'], batch['landmark3'])
        fake_23 = self.G(fake_12, batch['landmark3'])

        loss_G = self.criterion_G(batch['image1'], batch['image2'], fake_12, d_fake_12, fake_121, fake_13, fake_23)
        loss_G.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1, norm_type=2)
        
        return loss_G, fake_12


    def forward_D(self, batch):
        for p in self.D.parameters():
            p.requires_grad = True

        self.D.zero_grad()

        fake_12 = self.G(batch['image1'], batch['landmark2']).detach()
        fake_12.requires_grad = True

        d_fake_12 = self.D(fake_12)
        d_real_12 = self.D(batch['image2'])

        loss_D = self.criterion_D(self.D, d_fake_12, d_real_12, fake_12, batch['image2'])
        loss_D.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1, norm_type=2)

        return loss_D, d_real_12, d_fake_12


    def train(self):
        self.G.train()
        self.D.train()


    def eval(self):
        self.G.eval()
        self.D.eval()


    def load_model(self, model: Module, optimizer: Optimizer, scheduler: StepLR,  options: Options) -> Tuple[Module, Optimizer, StepLR, str, str]:
            filename = f'{type(model).__name__}_{options.continue_id}'
            state_dict = torch.load(os.path.join(options.checkpoint_dir, filename))
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['scheduler'])
            epoch = state_dict['epoch']
            iteration = state_dict['iteration']

            self.logger.log_info(f'Model loaded: {filename}')
            
            return model, optimizer, scheduler, epoch, iteration


    def save_model(self, model: Module, optimizer: Optimizer, scheduler: StepLR, epoch: str, iteration: str, options: Options, ext='.pth', time_for_name=None):
        if time_for_name is None:
            time_for_name = datetime.now()

        m = model.module if isinstance(model, DataParallel) else model
        # o = optimizer.module if isinstance(optimizer, DataParallel) else optimizer
        # s = scheduler.module if isinstance(scheduler, DataParallel) else scheduler

        m.eval()
        if options.device == 'cuda':
            m.cpu()

        filename = f'{type(m).__name__}_t_{time_for_name:%Y%m%d_%H%M}_e{str(epoch).zfill(2)}_i{str(iteration).zfill(7)}{ext}'
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
