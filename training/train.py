import logging
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from configs import TrainOptions
from testing import Test
from dataset import VoxCelebDataset
from dataset import Resize, RandomHorizontalFlip, RandomRotate, ToTensor
from models import Network, save_image

class Train():
    def __init__(self, options: TrainOptions):
        self.options = options
        self.training = True

        # Set seeds
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(57)
        torch.cuda.manual_seed(57)
        np.random.seed(57)

        self.data_loader_train = self._get_data_loader()
        self.network = Network(self.options, self.training)

        # Start training
        self()


    def _get_data_loader(self):
        transforms_list = [
            Resize(size=self.options.image_size),
            RandomHorizontalFlip() if self.options.horizontal_flip else None,
            RandomRotate(angle=self.options.angle) if self.options.angle > 0 else None,
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

        logging.info('===== TRAINING =====')
        logging.info(f'Running on {self.options.device.upper()}.')
        logging.info(f'----- OPTIONS -----')
        logging.info(self.options)
        logging.info(f'Epochs: {self.options.epochs} Batches/Iterations: {len(self.data_loader_train)} Batch Size: {self.options.batch_size}')

        for epoch in range(self.options.epochs):
            epoch_start =datetime.now()

            self._train(epoch)
            self._test(epoch)

            epoch_end = datetime.now()
            logging.info(f'Epoch {epoch + 1} finished in {epoch_end - epoch_start}.')

        run_end = datetime.now()
        logging.info(f'Training finished in {run_end - run_start}.')


    def _train(self, epoch):
        self.network.train()

        # TRAIN EPOCH
        self._train_epoch(epoch)

        # SAVE MODEL (EPOCH)
        self.network.save_model(self.network.G, self.options, self.run_start)
        self.network.save_model(self.network.D, self.options, self.run_start)
        
        self.network.scheduler_D.step()
        self.network.scheduler_G.step()


    def _train_epoch(self, epoch):
        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()

            # TODO: load d_iters from config
            d_iters = 5 if self.options.gan_type == 'wgan-gp' else 2
            if (batch_num + 1) % d_iters == 0:
                self.loss_G, self.image_fake = self.network.forward_G(batch)
                self.network.optimizer_G.step()
            else:
                self.loss_D, self.real, self.fake = self.network.forward_D(batch)
                self.network.optimizer_D.step()

            batch_end = datetime.now()

            # SHOW PROGRESS
            if (batch_num + 1) % 1 == 0 or batch_num == 0:
                logging.info(f'Epoch {epoch + 1}: [{batch_num + 1}/{len(self.data_loader_train)}] | '
                            f'Time: {batch_end - batch_start} | '
                            f'Loss_G = {self.loss_G.item():.4f} Loss_D = {self.loss_D.item():.4f}')
                logging.debug(f'D(real) = {self.real.mean().item():.4f} D(fake) = {self.fake.mean().item():.4f}')

            # LOG GENERATED IMAGES
            save_image(self.options.gen_dir, f'last_result_real.png', batch['image2'][0])
            save_image(self.options.gen_dir, f'last_result_fake.png', self.image_fake[0])

            if (batch_num + 1) % self.options.checkpoint_freq == 0:
                save_image(self.options.gen_dir, f'{datetime.now():%Y%m%d_%H%M%S%f}_real.png', epoch, batch_num, batch['image2'][0])
                save_image(self.options.gen_dir, f'{datetime.now():%Y%m%d_%H%M%S%f}_fake.png', epoch, batch_num, self.image_fake[0])

            # SAVE MODEL
            if (batch_num + 1) % self.options.checkpoint_freq == 0:
                self.network.save_model(self.network.G, self.options, self.run_start)
                self.network.save_model(self.network.D, self.options, self.run_start)


    def _test(self, epoch):
        test = Test(self.options, self.network, self.training)
        # Start testing
        test(epoch)
