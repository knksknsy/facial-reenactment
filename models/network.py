import os
from typing import Tuple
from datetime import datetime

import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer

from configs import Options
from models.generator import Generator, LossG
from models.discriminator import Discriminator, LossD
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
            self.continue_epoch = state_dict['epoch']

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
                betas=(self.options.beta1, self.options.beta2),
                weight_decay=self.options.weight_decay
            )

            self.optimizer_D = Adam(
                params=self.D.parameters(),
                lr=self.options.lr_d,
                betas=(self.options.beta1, self.options.beta2),
                weight_decay=self.options.weight_decay
            )

            if self.options.continue_id is not None:
                self.G, self.optimizer_G, self.continue_epoch, self.continue_iteration = self.load_model(self.G, self.optimizer_G, self.options)
                self.D, self.optimizer_D, self.continue_epoch, self.continue_iteration = self.load_model(self.D, self.optimizer_D, self.options)


    def __call__(self, images, landmarks):
        with torch.no_grad():
            return self.G(images, landmarks)


    def forward_G(self, batch):
        for p in self.D.parameters():
            p.requires_grad = False

        self.G.zero_grad()

        fake_12, fake_mask_12, _ = self.G(batch['image1'], batch['landmark2'])
        d_fake_12 = self.D(fake_12)
        fake_121, fake_mask_121, _ = self.G(fake_12, batch['landmark1'])
        fake_13, fake_mask_13, _ = self.G(batch['image1'], batch['landmark3'])
        fake_23, fake_mask_23, _ = self.G(fake_12, batch['landmark3'])

        loss_G, losses_dict = self.criterion_G(
            batch['image1'], batch['image2'], d_fake_12,
            fake_12, fake_121, fake_13, fake_23,
            fake_mask_12, fake_mask_121, fake_mask_13, fake_mask_23
        )
        loss_G.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.options.grad_clip, norm_type=2)

        self.optimizer_G.step()

        del fake_mask_12, d_fake_12, fake_121, fake_mask_121, fake_13, fake_mask_13, fake_23, fake_mask_23, _

        return fake_12.detach(), loss_G.detach().item(), losses_dict


    def get_loss_G(self, batch):
        with torch.no_grad():
            fake_12, fake_mask_12, _ = self.G(batch['image1'], batch['landmark2'])
            d_fake_12 = self.D(fake_12)
            fake_121, fake_mask_121, _ = self.G(fake_12, batch['landmark1'])
            fake_13, fake_mask_13, _ = self.G(batch['image1'], batch['landmark3'])
            fake_23, fake_mask_23, _ = self.G(fake_12, batch['landmark3'])

            loss_G, _ = self.criterion_G(
                batch['image1'], batch['image2'], d_fake_12,
                fake_12, fake_121, fake_13, fake_23,
                fake_mask_12, fake_mask_121, fake_mask_13, fake_mask_23
            )
            del fake_12, fake_mask_12, d_fake_12, fake_121, fake_mask_121, fake_13, fake_mask_13, fake_23, fake_mask_23, _
            return loss_G.detach().item()


    def forward_D(self, batch):
        for p in self.D.parameters():
            p.requires_grad = True

        self.D.zero_grad()

        fake_12, fake_mask_12, _ = self.G(batch['image1'], batch['landmark2'])
        fake_12 = fake_12.detach()
        fake_12.requires_grad = True

        d_fake_12 = self.D(fake_12)
        d_real_12 = self.D(batch['image2'])

        loss_D, losses_dict = self.criterion_D(self.D, d_fake_12, d_real_12, fake_12, batch['image2'])
        loss_D.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.options.grad_clip, norm_type=2)

        self.optimizer_D.step()

        del fake_12, fake_mask_12, _, d_real_12, d_fake_12

        return loss_D.detach().item(), losses_dict


    def get_loss_D(self, real, fake):
        d_fake = self.D(fake)
        d_real = self.D(real)
        loss_D, _ = self.criterion_D(self.D, d_fake, d_real, fake, real, req_grad=True)
        del d_fake, d_real, _
        return loss_D.detach().item()


    def train(self):
        self.G.train()
        self.D.train()


    def eval(self):
        self.G.eval()
        if self.model_path is None:
            self.D.eval()


    def load_model(self, model: Module, optimizer: Optimizer, options: Options) -> Tuple[Module, Optimizer, str, str]:
            filename = f'{type(model).__name__}_{options.continue_id}'
            state_dict = torch.load(os.path.join(options.checkpoint_dir, filename))
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            epoch = state_dict['epoch'] + 1
            iteration = state_dict['iteration']

            if options.overwrite_optim:
                optimizer.param_groups[0]['lr'] = options.lr_g if type(model).__name__ == 'Generator' else options.lr_d
                optimizer.param_groups[0]['betas'] = (options.beta1, options.beta2)
                optimizer.param_groups[0]['weight_decay'] = options.weight_decay

            self.logger.log_info(f'Model loaded: {filename}')
            
            return model, optimizer, epoch, iteration


    def save_model(self, model: Module, optimizer: Optimizer, epoch: str, iteration: str, options: Options, ext='.pth', time_for_name=None):
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
                'epoch': epoch,
                'iteration': iteration
            },
            os.path.join(options.checkpoint_dir, filename)
        )

        if options.device == 'cuda':
            m.to(options.device)
        m.train()

        self.logger.log_info(f'Model saved: {filename}')
