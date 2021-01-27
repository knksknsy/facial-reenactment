import os
import sys
import logging
from cv2 import cv2

import torch
from torch.nn import DataParallel
from torchvision import transforms
from torch.utils.data import DataLoader

from datetime import datetime

from configs import TrainOptions
from dataset import VoxCelebDataset
from dataset import Resize, RandomHorizontalFlip, RandomRotate, ToTensor
from network import Network

class Train():
    def __init__(self, options: TrainOptions):
        self.options = options

        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        self.data_loader_train, self.data_loader_test = self._get_data_loaders()
        self.network = Network(training=True, self.options)

        self._train()


    def _get_data_loaders(self):
        dataset_train = VoxCelebDataset(
            dataset_path=self.options.dataset_train_path,
            csv_file=self.options.csv_train,
            shuffle_frames=True,
            transform=transforms.Compose([
                        Resize(size=self.options.image_size),
                        RandomHorizontalFlip(),
                        RandomRotate(angle=self.options.angle),
                        ToTensor(device=self.options.device)
            ])
        )

        dataset_test = VoxCelebDataset(
            dataset_path=self.options.dataset_test_path,
            csv_file=self.options.csv_test,
            shuffle_frames=True,
            transform=transforms.Compose([
                        Resize(size=self.options.image_size),
                        ToTensor(device=self.options.device)
            ])
        )

        # num_workers, pin_memory
        data_loader_train = DataLoader(dataset_train, batch_size=self.options.batch_size, shuffle=True)
        data_loader_test = DataLoader(dataset_test, batch_size=self.options.batch_size, shuffle=True)

        return data_loader_train, data_loader_test


    def _train(self):
        self.run_start = datetime.now()

        logging.info('===== TRAINING =====')
        logging.info(f'Running on {self.options.device.upper()}.')

        logging.info(f'Training using dataset located in {self.options.dataset_train_path}')
        logging.info(f'Testing using dataset located in {self.options.dataset_test_path}')

        logging.info(f'Epochs: {self.options.epochs} Batches: {len(self.data_loader_train)} Batch Size: {self.options.batch_size}')

        for epoch in range(self.options.epochs):
            epoch_start = datetime.now()

            self.network.train()

            # TRAIN EPOCH
            self._train_epoch(epoch)

            # SAVE MODEL (EPOCH)
            # self.network.save_model(G, self.options, self.run_start)
            # self.network.save_model(D, self.options, self.run_start)
            
            self.network.scheduler_D.step()
            self.network.scheduler_G.step()

            epoch_end = datetime.now()
            logging.info(f'Epoch {epoch + 1} finished in {epoch_end - epoch_start}.')
        
        self.run_end = datetime.now()
        logging.info(f'Training finished in {self.run_end - self.run_start}.')


    def _train_epoch(self, epoch):

        for batch_num, (i, batch) in enumerate(self.data_loader_train):
            batch_start = datetime.now()

            # TODO: load d_iters from config
            d_iters = 5 if self.options.gan_type == 'wgan-gp' else 2
            if (batch_num + 1) % d_iters == 0:
                self.network.forward_G(batch)
                self.network.optimizer_G.step()
            else:
                self.network.forward_D(batch)
                self.network.optimizer_D.step()

            batch_end = datetime.now()

            # SHOW PROGRESS
            # if (batch_num + 1) % 1 == 0 or batch_num == 0:
                # logging.info(f'Epoch {epoch + 1}: [{batch_num + 1}/{len(data_loader_train)}] | '
                #              f'Time: {batch_end - batch_start} | '
                #              f'Loss_G = {loss_G.item():.4f} Loss_D = {loss_D.item():.4f}')
                # logging.debug(f'D(image) = {image.mean().item():.4f} D(image_hat) = {image_hat.mean().item():.4f}')

            # LOG GENERATED IMAGES
            # save_image(os.path.join(self.options.gen_dir, f'last_result_x.png'), image)
            # save_image(os.path.join(self.options.gen_dir, f'last_result_x_hat.png'), image_hat)

            # if (batch_num + 1) % 1000 == 0:
            #     save_image(os.path.join(self.options.gen_dir, f'{datetime.now():%Y%m%d_%H%M%S%f}_x.png'), image)
            #     save_image(os.path.join(self.options.gen_dir, f'{datetime.now():%Y%m%d_%H%M%S%f}_x_hat.png'), image_hat)

            # SAVE MODEL (N-ITERATIONS)
            # if (batch_num + 1) % 1000 == 0:
            #     self.network.save_model(G, gpu, self.run_start)
            #     self.network.save_model(D, gpu, self.run_start)


    def save_image(self, filename, data):
        if not os.path.isdir(self.options.gen_dir):
            os.makedirs(self.options.gen_dir)

        data = data.clone().detach().cpu()
        img = (data.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        cv2.imwrite(filename, img)


    def imshow(self, data):
        data = data.clone().detach().cpu()
        img = (data.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        cv2.imshow('Image Preview', img)
