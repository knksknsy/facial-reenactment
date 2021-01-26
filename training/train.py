import os
import sys
import logging
from cv2 import cv2

import torch
from torch.nn import DataParallel
from torchvision import transforms
from torch.utils.data import DataLoader

from datetime import datetime
from configs import Options

from dataset import VoxCelebDataset
from dataset import Resize, RandomHorizontalFlip, RandomRotate, ToTensor

class Train():
    def __init__(self, options: Options):
        self.options = options

        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        self.data_loader_train, self.data_loader_test = self._get_data_loaders()
        self.model = self._get_model()
        self._train()


    def _get_data_loaders(self):
        dataset_train = VoxCelebDataset(
            dataset_path=self.options.args.dataset_train_path,
            csv_file=self.options.args.csv_train,
            shuffle_frames=True,
            transform=transforms.Compose([
                        Resize(size=self.options.args.image_size),
                        RandomHorizontalFlip(),
                        RandomRotate(angle=self.options.args.angle),
                        ToTensor(device=self.options.device)
            ])
        )

        dataset_test = VoxCelebDataset(
            dataset_path=self.options.args.dataset_test_path,
            csv_file=self.options.args.csv_test,
            shuffle_frames=True,
            transform=transforms.Compose([
                        Resize(size=self.options.args.image_size),
                        ToTensor(device=self.options.device)
            ])
        )

        data_loader_train = DataLoader(dataset_train, batch_size=self.options.args.batch_size, shuffle=True)
        data_loader_test = DataLoader(dataset_test, batch_size=self.options.args.batch_size, shuffle=True)

        return data_loader_train, data_loader_test


    def _get_model(self):
        # TODO
        # G = Generator(self.options)
        # D = Discriminator(self.options)
        # criterion_G = LossG(self.options)
        # criterion_D = LossD(self.options)

        # optimizer_G = Adam(
        #     params=list(G.parameters()),
        #     lr=self.options.args.lr_g
        # )
        # optimizer_D = Adam(
        #     params=D.parameters(),
        #     lr=self.options.args.lr_d
        # )

        # if continue_id is not None:
        #     G = load_model(G, continue_id)
        #     D = load_model(D, continue_id)
        
        return 1


    def _train(self):
        self.run_start = datetime.now()

        logging.info('===== TRAINING =====')
        logging.info(f'Running on {self.options.device.upper()}.')

        logging.info(f'Training using dataset located in {self.options.args.dataset_train_path}')
        logging.info(f'Testing using dataset located in {self.options.args.dataset_test_path}')

        logging.info(f'Epochs: {self.options.args.epochs} Batches: {len(self.data_loader_train)} Batch Size: {self.options.args.batch_size}')

        for epoch in range(self.options.args.epochs):
            epoch_start = datetime.now()

            # TRAIN EPOCH
            self._train_epoch(epoch)

            # SAVE MODEL (EPOCH)
            # save_model(G, self.options, self.run_start)
            # save_model(D, self.options, self.run_start)

            epoch_end = datetime.now()
            logging.info(f'Epoch {epoch + 1} finished in {epoch_end - epoch_start}.')
        
        self.run_end = datetime.now()
        logging.info(f'Training finished in {self.run_end - self.run_start}.')


    def _train_epoch(self, epoch):
        # G.train()
        # D.train()

        for batch_num, (i, samples) in enumerate(self.data_loader_train):
            batch_start = datetime.now()

            # LOAD BATCH
            image1, image2, image3 = samples['image1'], samples['image2'], samples['image3']
            landmark1, landmark2, landmark3 = samples['landmark1'], samples['landmark2'], samples['landmark3']

            # FORWARD
            # image_hat = G(...)
            # r_image_hat = D(image_hat, ...)
            # r_image = D(image, ...)


            # OPTIMIZE
            # ...
            # optimizer_G.zero_grad()
            # optimizer_D.zero_grad()
            
            # loss_G = criterion_G(...)
            # loss_D = criterion_D(...)
            # loss = loss_G + loss_D
            # loss.backward()

            # optimizer_G.step()
            # optimizer_D.step()
            # ...

            batch_end = datetime.now()

            # SHOW PROGRESS
            # if (batch_num + 1) % 1 == 0 or batch_num == 0:
                # logging.info(f'Epoch {epoch + 1}: [{batch_num + 1}/{len(data_loader_train)}] | '
                #              f'Time: {batch_end - batch_start} | '
                #              f'Loss_G = {loss_G.item():.4f} Loss_D = {loss_D.item():.4f}')
                # logging.debug(f'D(image) = {image.mean().item():.4f} D(image_hat) = {image_hat.mean().item():.4f}')

            # LOG GENERATED IMAGES
            # save_image(os.path.join(self.options.args.gen_dir, f'last_result_x.png'), image)
            # save_image(os.path.join(self.options.args.gen_dir, f'last_result_x_hat.png'), image_hat)

            # if (batch_num + 1) % 1000 == 0:
            #     save_image(os.path.join(self.options.args.gen_dir, f'{datetime.now():%Y%m%d_%H%M%S%f}_x.png'), image)
            #     save_image(os.path.join(self.options.args.gen_dir, f'{datetime.now():%Y%m%d_%H%M%S%f}_x_hat.png'), image_hat)

            # SAVE MODEL (N-ITERATIONS)
            # if (batch_num + 1) % 1000 == 0:
            #     save_model(G, gpu, self.run_start)
            #     save_model(D, gpu, self.run_start)


    # TODO: Move probably to Model-Class
    def save_model(self, model, device, time_for_name=None):
        if time_for_name is None:
            time_for_name = datetime.now()

        m = model.module if isinstance(model, DataParallel) else model

        m.eval()
        if device == 'cuda':
            m.cpu()

        if not os.path.exists(self.options.args.models_dir):
            os.makedirs(self.options.args.models_dir)
        filename = f'{type(m).__name__}_{time_for_name:%Y%m%d_%H%M}.pth'
        torch.save(
            m.state_dict(),
            os.path.join(self.options.args.models_dir, filename)
        )

        if device == 'cuda':
            m.to(device)
        m.train()

        logging.info(f'Model saved: {filename}')


    def load_model(self, model, continue_id):
        filename = f'{type(model).__name__}_{continue_id}.pth'
        state_dict = torch.load(os.path.join(self.options.args.models_dir, filename))
        model.load_state_dict(state_dict)
        return model


    def save_image(self, filename, data):
        if not os.path.isdir(self.options.args.gen_dir):
            os.makedirs(self.options.args.gen_dir)

        data = data.clone().detach().cpu()
        img = (data.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        cv2.imwrite(filename, img)


    def imshow(self, data):
        data = data.clone().detach().cpu()
        img = (data.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        cv2.imshow('Image Preview', img)
