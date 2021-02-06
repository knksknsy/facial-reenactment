import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from configs import TrainOptions
from testing import Test
from dataset import VoxCelebDataset
from dataset import Resize, RandomHorizontalFlip, RandomRotate, ToTensor
from models.network import Network
from logger import Logger

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

        self.data_loader_train = self._get_data_loader()
        self.network = Network(self.logger, self.options, self.training)

        # Start training
        self()


    def _get_data_loader(self):
        transforms_list = [
            Resize(size=self.options.image_size),
            RandomHorizontalFlip() if self.options.horizontal_flip else None,
            RandomRotate(angle=self.options.rotation_angle) if self.options.rotation_angle > 0 else None,
            ToTensor(device=self.options.device),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
        compose = [t for t in transforms_list if t is not None]

        dataset_train = VoxCelebDataset(
            dataset_path=self.options.dataset_train,
            csv_file=self.options.csv_train,
            shuffle_frames=self.options.shuffle_frames,
            transform=transforms.Compose(compose),
            training=self.training
        )

        data_loader_train = DataLoader(dataset_train,
                                        batch_size=self.options.batch_size,
                                        shuffle=self.options.shuffle,
                                        num_workers=self.options.num_workers,
                                        pin_memory=self.options.pin_memory
        )

        return data_loader_train


    def __call__(self):
        run_start = datetime.now()

        self.iterations = self.network.continue_iteration

        self.logger.log_info('===== TRAINING =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')
        self.logger.log_info(f'----- OPTIONS -----')
        self.logger.log_info(self.options)
        self.logger.log_info(f'Epochs: {self.options.epochs} Batches/Iterations: {len(self.data_loader_train)} Batch Size: {self.options.batch_size}')

        for epoch in range(self.network.continue_epoch, self.options.epochs):
            epoch_start =datetime.now()

            self._train(epoch)
            self._test(epoch)

            epoch_end = datetime.now()
            self.logger.log_info(f'Epoch {epoch + 1} finished in {epoch_end - epoch_start}.')

        run_end = datetime.now()
        self.logger.log_info(f'Training finished in {run_end - run_start}.')


    def _train(self, epoch):
        self.network.train()

        # TRAIN EPOCH
        self._train_epoch(epoch)
        
        self.network.scheduler_D.step()
        self.network.scheduler_G.step()

        # SAVE MODEL (EPOCH)
        self.network.save_model(self.network.G, self.network.optimizer_G, self.network.scheduler_G, epoch, self.iterations, self.options, self.run_start)
        self.network.save_model(self.network.D, self.network.optimizer_D, self.network.scheduler_D, epoch, self.iterations, self.options, self.run_start)


    def _train_epoch(self, epoch):
        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()

            d_iters = 5 if self.options.gan_type == 'wgan-gp' else 2
            if (batch_num + 1) % d_iters == 0:
                self.loss_G, self.image_fake = self.network.forward_G(batch)
                self.network.optimizer_G.step()
            else:
                self.loss_D, self.d_real, self.d_fake = self.network.forward_D(batch)
                self.network.optimizer_D.step()

            batch_end = datetime.now()

            # SHOW PROGRESS
            if (batch_num + 1) % 1 == 0 or batch_num == 0:
                self.logger.log_info(f'Epoch {epoch + 1}: [{batch_num + 1}/{len(self.data_loader_train)}] | '
                                    f'Time: {batch_end - batch_start} | '
                                    f'Loss_G = {self.loss_G.item():.4f} Loss_D = {self.loss_D.item():.4f}')
                self.logger.log_debug(f'D(real) = {self.d_real.mean().item():.4f} D(fake) = {self.d_fake.mean().item():.4f}')
                self.logger.log_scalar('Loss_G', self.loss_G.item(), self.iterations)
                self.logger.log_scalar('Loss_D', self.loss_D.item(), self.iterations)
                self.logger.log_scalar('D(real)', self.d_real.mean().item(), self.iterations)
                self.logger.log_scalar('D(fake)', self.d_fake.mean().item(), self.iterations)

            # LOG GENERATED IMAGES
            images_real = batch['image2'].clone().detach()
            images_fake = self.image_fake.clone().detach()
            images = torch.cat((images_real, images_fake), dim=1)
            self.logger.save_image(self.options.gen_dir, f'0_last_result.png', images)

            if (batch_num + 1) % self.options.log_freq == 0:
                self.logger.save_image(self.options.gen_dir, f'{datetime.now():%Y%m%d_%H%M%S%f}.png', images, epoch, self.iterations)
                self.logger.log_image('Train/Generated', images, self.iterations)

            # SAVE MODEL
            if (batch_num + 1) % self.options.checkpoint_freq == 0:
                self.network.save_model(self.network.G, self.network.optimizer_G, self.network.scheduler_G, epoch, self.iterations, self.options, self.run_start)
                self.network.save_model(self.network.D, self.network.optimizer_D, self.network.scheduler_D, epoch, self.iterations, self.options, self.run_start)

            self.iterations += 1


    def _test(self, epoch):
        test = Test(self.logger, self.options, self.network, self.training)
        # Start testing
        test(epoch)
