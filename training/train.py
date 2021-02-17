import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from configs.train_options import TrainOptions
from testing.test import Test
from testing.ssim import calculate_ssim
from testing.fid import FrechetInceptionDistance
from dataset.dataset import VoxCelebDataset
from dataset.transforms import Resize, RandomHorizontalFlip, RandomRotate, ToTensor, Normalize
from models.network import Network
from models.utils import lr_linear_scheduler
from loggings.logger import Logger

class Train():
    def __init__(self, logger: Logger, options: TrainOptions):
        self.logger = logger
        self.options = options
        self.training = True

        # Set seeds
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(57)
        torch.cuda.manual_seed(57)
        np.random.seed(57)

        self.data_loader_train = self._get_data_loader(train_format=self.training)
        self.network = Network(self.logger, self.options, model_path=None)
        self.fid = FrechetInceptionDistance(self.options, device='cpu', data_loader_length=1)

        # Start training
        self()


    def _get_data_loader(self, train_format):
        transforms_list = [
            Resize(self.options.image_size, train_format),
            RandomHorizontalFlip(train_format) if self.options.horizontal_flip else None,
            RandomRotate(self.options.rotation_angle, train_format) if self.options.rotation_angle > 0 else None,
            ToTensor(self.options.device, train_format),
            Normalize(0.5, 0.5, train_format)
        ]
        compose = [t for t in transforms_list if t is not None]

        dataset_train = VoxCelebDataset(
            self.options.dataset_train,
            self.options.csv_train,
            self.options.shuffle_frames,
            transforms.Compose(compose),
            train_format
        )

        data_loader_train = DataLoader(
            dataset_train,
            self.options.batch_size,
            self.options.shuffle,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory
        )

        return data_loader_train


    def __call__(self):
        self.run_start = datetime.now()

        self.iterations = self.network.continue_iteration

        self.logger.log_info('===== TRAINING =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')
        self.logger.log_info(f'----- OPTIONS -----')
        self.logger.log_info(self.options)
        self.logger.log_info(f'Epochs: {self.options.epochs} Batches/Iterations: {len(self.data_loader_train)} Batch Size: {self.options.batch_size}')
        self.logger.log_info(f'Learning Rates: Generator = {self.network.optimizer_G.param_groups[0]["lr"]} Discriminator = {self.network.optimizer_D.param_groups[0]["lr"]}')

        for epoch in range(self.network.continue_epoch, self.options.epochs):
            epoch_start =datetime.now()

            self._train(epoch)

            if self.options.test:
                Test(self.logger, self.options, self.network).test(epoch)

            # TODO: create new event file
            self.logger.init_writer()

            epoch_end = datetime.now()
            self.logger.log_info(f'Epoch {epoch + 1} finished in {epoch_end - epoch_start}.')

        self.run_end = datetime.now()
        self.logger.log_info(f'Training finished in {self.run_end - self.run_start}.')


    def _train(self, epoch):
        self.network.train()

        # TRAIN EPOCH
        self._train_epoch(epoch)
        
        # Schedule learning rate
        self.network.optimizer_G.param_groups[0]['lr'] = lr_linear_scheduler(
            epoch,
            epoch_start=self.options.scheduler_epoch_range[0],
            epoch_end=self.options.scheduler_epoch_range[1],
            lr_base=self.options.lr_g,
            lr_end=self.options.scheduler_lr_g_end
        )
        self.network.optimizer_D.param_groups[0]['lr'] = lr_linear_scheduler(
            epoch,
            epoch_start=self.options.scheduler_epoch_range[0],
            epoch_end=self.options.scheduler_epoch_range[1],
            lr_base=self.options.lr_d,
            lr_end=self.options.scheduler_lr_d_end
        )
        # self.network.scheduler_D.step()
        # self.network.scheduler_G.step()

        # LOG LEARNING RATES
        self.logger.log_info(f'Update learning rates: LR_G = {self.network.optimizer_G.param_groups[0]["lr"]:.8f} LR D = {self.network.optimizer_D.param_groups[0]["lr"]:.8f}')
        self.logger.log_scalar('LR_G', self.network.optimizer_G.param_groups[0]["lr"], epoch)
        self.logger.log_scalar('LR_D', self.network.optimizer_D.param_groups[0]["lr"], epoch)

        # SAVE MODEL (EPOCH)
        self.network.save_model(self.network.G, self.network.optimizer_G, self.network.scheduler_G, epoch, self.iterations, self.options, time_for_name=self.run_start)
        self.network.save_model(self.network.D, self.network.optimizer_D, self.network.scheduler_D, epoch, self.iterations, self.options, time_for_name=self.run_start)


    def _train_epoch(self, epoch):
        gen_call_counter = 0

        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()
            is_gen_active = False


            # Adaptive update strategy
            if self.options.loss_coeff > 0:
                # First iteration: initialize loss change ratios r_d, r_g
                if epoch == self.network.continue_epoch and batch_num == 0:
                    self.r_d = self.r_g = 1
                
                if self.r_d > self.options.loss_coef * self.r_g:
                    is_gen_active = False
                    self.loss_D, self.d_real, self.d_fake = self.network.forward_D(batch, self.iterations)
                else:
                    is_gen_active = True
                    gen_call_counter += 1
                    self.image_fake, self.loss_G = self.network.forward_G(batch, self.iterations)
                
                # First iteration: set current losses to previous losses
                if epoch == self.network.continue_epoch and batch_num == 0:
                    self.loss_p_g, self.loss_p_d = self.loss_G, self.loss_D
                
                self.loss_c_g, self.loss_c_d = self.loss_G, self.loss_D
                self.r_g = abs((self.loss_c_g - self.loss_p_g) / self.loss_p_g)
                self.r_d = abs((self.loss_c_d - self.loss_p_d) / self.loss_p_d)
                self.loss_p_g, self.loss_p_d = self.loss_G, self.loss_D


            # Fixed update strategy
            elif self.options.d_iters > 0:
                d_iters = self.options.d_iters if self.options.gan_type == 'wgan-gp' else 2
                is_gen_active = (batch_num + 1) % d_iters == 0
                
                if is_gen_active:
                    self.image_fake, self.loss_G = self.network.forward_G(batch, self.iterations)
                else:
                    self.loss_D, self.d_real, self.d_fake = self.network.forward_D(batch, self.iterations)

            batch_end = datetime.now()


            # LOG PROGRESS
            if is_gen_active:
                self.logger.log_info(f'Epoch {epoch + 1}: [{str(batch_num + 1).zfill(len(str(len(self.data_loader_train))))}/{len(self.data_loader_train)}] | '
                                    f'Time: {batch_end - batch_start} | '
                                    f'Loss_G = {self.loss_G.item():.4f} Loss_D = {self.loss_D.item():.4f} | '
                                    f'D(real) = {self.d_real.mean().item():.4f} D(fake) = {self.d_fake.mean().item():.4f}')

                # LOG LATEST GENERATED IMAGE
                images_real = batch['image2'].detach().clone()
                images_fake = self.image_fake.detach().clone()
                images = torch.cat((images_real, images_fake), dim=0)
                self.logger.save_image(self.options.gen_dir, f'0_last_result', images)
                del images_real, images_fake

                # LOG GENERATED IMAGES
                if (batch_num + 1) % self.options.log_freq == 0 or gen_call_counter % self.options.log_freq == 0:
                    gen_call_counter = 0
                    self.logger.save_image(self.options.gen_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=self.iterations)
                    self.logger.log_image('Train/Generated', images, self.iterations)
                    del images
                    
                    # LOG EVALUATION METRICS
                    if self.options.test:
                        val_time_start = datetime.now()
                        images_real = batch['image2'].detach().clone()
                        images_fake = self.image_fake.detach().clone()
                        ssim_train, fid_train = self.evaluate_metrics(images_real, images_fake, device=self.fid.device)
                        val_time_end = datetime.now()
                        self.logger.log_info(f'Validation: Time: {val_time_end - val_time_start} | SSIM = {ssim_train:.4f} | FID = {fid_train:.4f}')
                        self.logger.log_scalar('SSIM Train', ssim_train, self.iterations)
                        self.logger.log_scalar('FID Train', fid_train, self.iterations)
                        del images_real, images_fake
            
                del self.image_fake, self.loss_G, self.loss_D, self.d_real, self.d_fake

            # # SAVE MODEL
            # if (batch_num + 1) % self.options.checkpoint_freq == 0:
            #     self.network.save_model(self.network.G, self.network.optimizer_G, self.network.scheduler_G, epoch, self.iterations, self.options, self.run_start)
            #     self.network.save_model(self.network.D, self.network.optimizer_D, self.network.scheduler_D, epoch, self.iterations, self.options, self.run_start)

            self.iterations += 1


    def evaluate_metrics(self, images_real, images_fake, device):
        ssim_score = calculate_ssim(images_fake.to(device), images_real.to(device))

        self.fid.calculate_activations(images_real.to(device), images_fake.to(device), batch_num=1)
        fid_score = self.fid.calculate_fid()

        return ssim_score.mean().item(), fid_score
