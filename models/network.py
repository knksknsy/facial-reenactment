import os
import sys
import logging
from datetime import datetime

import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from configs import Options
from models.generator import Generator, LossG
from models.discriminator import Discriminator, LossD

class Network():
    def __init__(self, options: Options, training=False):
        self.training = training
        self.options = options

        self.G = Generator(self.options)

        # Training mode
        if self.training:
            self.D = Discriminator(self.options)

            # Load networks into multiple GPUs
            if torch.cuda.device_count() > 1:
                self.G = DataParallel(self.G)
                self.D = DataParallel(self.D)

            if self.options.continue_id:
                self.G = self.load_model(self.G, self.options)
                self.D = self.load_model(self.D, self.options)

            self.criterion_G = LossG(self.options)
            self.criterion_D = LossD(self.options)

            # TODO: implement architectures: generator, discriminator
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
                step_size=self.options.step_size,
                gamma=self.options.gamma
            )

            self.scheduler_D = StepLR(
                optimizer=self.optimizer_D,
                step_size=self.options.step_size,
                gamma=self.options.gamma
            )

        # Testing mode
        else:
            state_dict = torch.load(self.options.model)
            self.G.load_state_dict(state_dict)


    def __call__(self, images, landmarks):
        with torch.no_grad():
            return self.G(images, landmarks)


    def forward_G(self, batch):
        for p in self.D.parameters():
            p.requires_grad = False

        self.G.zero_grad()

        AB = self.G(batch['image1'], batch['landmark2'])
        pred_AB = self.D(AB)
        ABA = self.G(AB, batch['landmark1'])
        AC = self.G(batch['image1'], batch['landmark3'])
        BC = self.G(AB, batch['landmark3'])

        loss_G = self.criterion_G(batch['image1'], batch['image2'], AB, pred_AB, ABA, AC, BC)
        loss_G.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1, norm_type=2)
        
        return loss_G, AB


    def forward_D(self, batch):
        for p in self.D.parameters():
            p.requires_grad = True

        self.D.zero_grad()

        AB = self.G(batch['image1'], batch['landmark2']).detach()
        AB.requires_grad = True

        fake_AB = self.D(AB)
        real_AB = self.D(batch['image2'])

        loss_D = self.criterion_D(self.D, fake_AB, real_AB, AB, batch['image2'])
        loss_D.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1, norm_type=2)

        return loss_D, real_AB, fake_AB


    def train(self):
        self.G.train()
        self.D.train()


    def eval(self):
        self.G.eval()
        self.D.eval()


    def load_model(self, model, options: Options):
            filename = f'{type(model).__name__}_{options.continue_id}.pth'
            state_dict = torch.load(os.path.join(options.models_dir, filename))
            model.load_state_dict(state_dict)

            logging.info(f'Model loaded: {filename}')
            
            return model


    def save_model(self, model, options: Options, time_for_name=None):
        if time_for_name is None:
            time_for_name = datetime.now()

        m = model.module if isinstance(model, DataParallel) else model

        m.eval()
        if options.device == 'cuda':
            m.cpu()

        if not os.path.exists(options.models_dir):
            os.makedirs(options.models_dir)

        filename = f'{type(m).__name__}_{time_for_name:%Y%m%d_%H%M}.pth'
        torch.save(
            m.state_dict(),
            os.path.join(options.models_dir, filename)
        )

        if options.device == 'cuda':
            m.to(options.device)
        m.train()

        logging.info(f'Model saved: {filename}')
