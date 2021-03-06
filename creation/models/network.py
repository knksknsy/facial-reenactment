import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from typing import Tuple

from configs.options import Options
from utils.models import load_model, save_model
from loggings.logger import Logger
from ..models.generator import Generator, LossG
from ..models.discriminator import Discriminator, LossD

class Network():
    def __init__(self, logger: Logger, options: Options, model_path=None):
        self.logger = logger
        self.model_path = model_path
        self.options = options

        self.continue_epoch = 0
        self.continue_iteration = 0

        # Testing mode
        if self.model_path is not None:
            # Load Generator
            self.G = Generator(self.options)
            state_dict_G = torch.load(self.model_path, map_location=torch.device('cpu') if self.options.device == 'cpu' else None)
            self.G.load_state_dict(state_dict_G['model'])
            self.continue_epoch = state_dict_G['epoch']
            # Load Discriminator
            self.D = Discriminator(self.options)
            state_dict_D = torch.load(self.model_path.replace('Generator', 'Discriminator'), map_location=torch.device('cpu') if self.options.device == 'cpu' else None)
            self.D.load_state_dict(state_dict_D['model'])

            self.criterion_G = LossG(self.logger, self.options, vgg_device=self.options.device, lightcnn_device=self.options.device)
            self.criterion_D = LossD(self.logger, self.options)

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

            self.criterion_G = LossG(self.logger, self.options, vgg_device=self.options.device, lightcnn_device=self.options.device)
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

            # Step based LR decay schedule
            if 'lr_step_decay' in self.options.config['train']['optimizer']:
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
            # Plateau based LR decay schedule
            elif 'lr_plateau_decay' in self.options.config['train']['optimizer']:
                self.scheduler_G = ReduceLROnPlateau(
                    optimizer=self.optimizer_G,
                    mode=self.options.plateau_mode,
                    factor=self.options.plateau_factor,
                    patience=self.options.plateau_patience,
                    cooldown=2*self.options.plateau_patience,
                    min_lr=self.options.plateau_min_lr_g
                )
                self.scheduler_D = ReduceLROnPlateau(
                    optimizer=self.optimizer_D,
                    mode=self.options.plateau_mode,
                    factor=self.options.plateau_factor,
                    patience=self.options.plateau_patience,
                    cooldown=2*self.options.plateau_patience,
                    min_lr=self.options.plateau_min_lr_d
                )
            else:#elif 'lr_linear_decay' in self.options.config['train']['optimizer']:
                self.scheduler_G, self.scheduler_D = None, None

            if self.options.continue_id is not None:
                self.G, self.optimizer_G, self.scheduler_G, self.continue_epoch, self.continue_iteration = self.load_model(self.G, self.optimizer_G, self.scheduler_G, self.options)
                self.D, self.optimizer_D, self.scheduler_D, self.continue_epoch, self.continue_iteration = self.load_model(self.D, self.optimizer_D, self.scheduler_D, self.options)


    def __call__(self, images, landmarks, batch = None, calc_loss:bool=False):
        # During inference
        if not calc_loss:
            with torch.no_grad():
                return self.G(images, landmarks)
        # During testing
        else:
            image_generated, loss_G, losses_G_dict = self.get_loss_G(batch)
            loss_D, losses_D_dict = self.get_loss_D(batch['image2'], image_generated)
            return image_generated, loss_G, losses_G_dict, loss_D, losses_D_dict


    def forward_G(self, batch):
        for p in self.D.parameters():
            p.requires_grad = False

        self.G.zero_grad()

        fake_12 = self.G(batch['image1'], batch['landmark2'])
        fm_fake_12, d_fake_12 = self.D(fake_12)
        fm_real_2, d_real_2 = self.D(batch['image2'])
        fake_121 = self.G(fake_12, batch['landmark1'])
        fake_13 = self.G(batch['image1'], batch['landmark3'])
        fake_23 = self.G(fake_12, batch['landmark3'])

        loss_G, losses_dict = self.criterion_G(
            batch['image1'], batch['image2'], d_fake_12,
            fake_12, fake_121, fake_13, fake_23,
            fm_fake_12, fm_real_2
        )
        loss_G.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.options.grad_clip, norm_type=2)

        self.optimizer_G.step()

        del d_fake_12, fake_121, fake_13, fake_23, d_real_2, fm_fake_12, fm_real_2
        fake_12 = fake_12.detach()
        loss_G = loss_G.detach().item()
        return fake_12, loss_G, losses_dict


    def get_loss_G(self, batch):
        with torch.no_grad():
            fake_12 = self.G(batch['image1'], batch['landmark2'])
            fm_fake_12, d_fake_12 = self.D(fake_12)
            fm_real_2, d_real_2 = self.D(batch['image2'])
            fake_121= self.G(fake_12, batch['landmark1'])
            fake_13 = self.G(batch['image1'], batch['landmark3'])
            fake_23 = self.G(fake_12, batch['landmark3'])

            loss_G, losses_G_dict = self.criterion_G(
                batch['image1'], batch['image2'], d_fake_12,
                fake_12, fake_121, fake_13, fake_23,
                fm_fake_12, fm_real_2
            )
            del d_fake_12, fake_121, fake_13, fake_23, d_real_2, fm_fake_12, fm_real_2
            loss_G = loss_G.detach().item()
            return fake_12, loss_G, losses_G_dict


    def forward_D(self, batch):
        for p in self.D.parameters():
            p.requires_grad = True

        self.D.zero_grad()

        fake_12 = self.G(batch['image1'], batch['landmark2'])
        fake_12 = fake_12.detach()
        fake_12.requires_grad = True

        fm_fake_12, d_fake_12 = self.D(fake_12)
        fm_real_2, d_real_2 = self.D(batch['image2'])

        loss_D, losses_dict = self.criterion_D(self.D, d_fake_12, d_real_2, fake_12, batch['image2'])
        loss_D.backward()

        if self.options.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.options.grad_clip, norm_type=2)

        self.optimizer_D.step()

        # Clamp parameters if in gradient clipping mode
        if self.options.l_gc > 0:
            for p in self.D.parameters():
                p.data.clamp_(-self.options.l_gc, self.options.l_gc)

        del fake_12, d_real_2, d_fake_12, fm_fake_12, fm_real_2
        loss_D = loss_D.detach().item()
        return loss_D, losses_dict


    def get_loss_D(self, real, fake):
        with torch.no_grad():
            fm_fake, d_fake = self.D(fake)
            fm_real, d_real = self.D(real)
            loss_D, losses_D_dict = self.criterion_D(self.D, d_fake, d_real, fake, real, skip_gp=True)
            del d_fake, d_real, fm_fake, fm_real
            loss_D = loss_D.detach().item()
            return loss_D, losses_D_dict


    def train(self):
        self.G.train()
        self.D.train()


    def eval(self):
        self.G.eval()
        if self.model_path is None:
            self.D.eval()


    def load_model(self, model: Module, optimizer: Optimizer, scheduler: object, options: Options) -> Tuple[Module, Optimizer, object, str, str]:
        filename = f'{type(model).__name__}_{options.continue_id}'
        self.logger.log_info(f'Model loaded: {filename}')
        return load_model(model, optimizer, scheduler, options)


    def save_model(self, model: Module, optimizer: Optimizer, scheduler: object, epoch: str, iteration: str, options: Options, ext='.pth', time_for_name=None):
        filename = f'{type(model).__name__}_t{time_for_name:%Y%m%d_%H%M}_e{str(epoch).zfill(3)}_i{str(iteration).zfill(8)}{ext}'
        save_model(model, optimizer, scheduler, epoch, iteration, options, ext, time_for_name)
        self.logger.log_info(f'Model saved: {filename}')
