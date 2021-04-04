import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from configs.train_options import TrainOptions
from testing.detection_test import TesterDetection
from dataset.dataset import FaceForensicsDataset
from dataset.faceforensics_transforms import Resize, GrayScale, RandomHorizontalFlip, RandomRotate, ToTensor, Normalize
from models.detection_network import NetworkDetection
from models.utils import lr_linear_schedule, init_seed_state
from loggings.logger import Logger


class TrainerDetection():
    def __init__(self, logger: Logger, options: TrainOptions):
        self.logger = logger
        self.options = options
        self.training = True

        torch.backends.cudnn.benchmark = True
        init_seed_state(self.options)

        self.data_loader_train = self._get_data_loader()
        # TODO: implement
        self.network = NetworkDetection(self.logger, self.options, model_path=None)

        # Start training
        self()


    def _get_data_loader(self):
        transforms_list = [
            Resize(self.options.image_size),
            GrayScale() if self.options.channels <= 1 else None,
            RandomHorizontalFlip() if self.options.horizontal_flip else None,
            RandomRotate(self.options.rotation_angle) if self.options.rotation_angle > 0 else None,
            ToTensor(self.options.channels, self.options.device),
            Normalize(self.options.normalize[0], self.options.normalize[1])
        ]
        transforms_list = [t for t in transforms_list if t is not None]

        dataset_train = FaceForensicsDataset(
            self.options.dataset_train,
            self.options.csv_train,
            self.options.image_size,
            self.options.channels,
            transforms.Compose(transforms_list)
        )

        # TODO: batch size / 2
        data_loader_train = DataLoader(
            dataset_train,
            self.options.batch_size,
            self.options.shuffle,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory,
            drop_last=True
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
        self.logger.log_info(f'Learning Rate = {self.network.optimizer.param_groups[0]["lr"]}')

        for epoch in range(self.network.continue_epoch, self.options.epochs):
            epoch_start = datetime.now()

            self._train(epoch)

            if self.options.test:
                # TODO: implement
                monitor_val = TesterDetection(self.logger, self.options, self.network).test(epoch)
                # Decrease LR if monitor_val stagnates
                if 'lr_plateau_decay' in self.options.config['train']['optimizer']:
                    self.network.scheduler.step(monitor_val)

            # Create new tensorboard event for each epoch
            self.logger.init_writer(filename_suffix=f'.{str(epoch+1)}')

            epoch_end = datetime.now()
            self.logger.log_info(f'Epoch {epoch + 1} finished in {epoch_end - epoch_start}.')

        self.run_end = datetime.now()
        self.logger.log_info(f'Training finished in {self.run_end - self.run_start}.')


    def _train(self, epoch):
        self.network.train()

        # TRAIN EPOCH
        self._train_epoch(epoch)

        # Schedule learning rate
        if 'lr_linear_decay' in self.options.config['train']['optimizer']:
            self.network.optimizer.param_groups[0]['lr'] = lr_linear_schedule(
                epoch,
                epoch_start=self.options.epoch_range[0],
                epoch_end=self.options.epoch_range[1],
                lr_base=self.options.lr_g,
                lr_end=self.options.lr_g_end
            )
        elif 'lr_step_decay' in self.options.config['train']['optimizer']:
            self.network.scheduler.step()

        # LOG LEARNING RATES
        self.logger.log_info(f'Update learning rate: LR = {self.network.optimizer.param_groups[0]["lr"]:.8f}')
        self.logger.log_scalar('LR', self.network.optimizer.param_groups[0]["lr"], epoch)

        # SAVE MODEL (EPOCH)
        self.network.save_model(self.network, self.network.optimizer, self.network.scheduler, epoch, self.iterations, self.options, time_for_name=self.run_start)


    def _train_epoch(self, epoch):
        loss = None

        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()

            d_iters = self.options.d_iters
            preds, features, loss, losses_dict = self.network.forward(batch)

            batch_end = datetime.now()

            # LOG PROGRESS
            if loss is not None and (batch_num + 1) % d_iters == 0:
                cur_it = str(batch_num + 1).zfill(len(str(len(self.data_loader_train))))
                total_it = len(self.data_loader_train) if self.options.iterations == 0 else self.options.iterations

                self.logger.log_info(f'Epoch {epoch + 1}: [{cur_it}/{total_it}] | '
                                    f'Time: {batch_end - batch_start} | '
                                    f'Loss = {loss:.4f}')

                # LOG LOSSES G AND D
                self.logger.log_scalars(losses_dict, self.iterations)

                # LOG LATEST FEATURES
                images_real = batch['images_real'].detach().clone()
                images_fake = batch['images_fake'].detach().clone()
                images_features = features.detach().clone()
                images = torch.cat((images_real, images_fake, images_features), dim=0)
                self.logger.save_image(self.options.gen_dir, f'0_last_result', images, nrow=self.options.batch_size)

                # LOG FEATURES
                if (batch_num + 1) % self.options.log_freq == 0:
                    self.logger.save_image(self.options.gen_dir, f't_{datetime.now():%Y%m%d_%H%M%S}',
                                            images, epoch=epoch, iteration=self.iterations, nrow=self.options.batch_size)
                    self.logger.log_image('Train/Features', images, self.iterations, nrow=self.options.batch_size)

                    # LOG EVALUATION METRICS
                    if self.options.metrics:
                        # val_time_start = datetime.now()
                        # ssim_train, fid_train = self.evaluate_metrics(images_real, images_fake, self.fid.device)
                        # val_time_end = datetime.now()
                        # self.logger.log_info(f'Validation: Time: {val_time_end - val_time_start} | SSIM = {ssim_train:.4f} | FID = {fid_train:.4f}')
                        # self.logger.log_scalar('SSIM Train', ssim_train, self.iterations)
                        # self.logger.log_scalar('FID Train', fid_train, self.iterations)
                        del images_real, images_fake, images#, ssim_train, fid_train
                    else:
                        del images_real, images_fake, images
                else:
                    del images_real, images_fake, images

            # # SAVE MODEL
            # if (batch_num + 1) % self.options.checkpoint_freq == 0:
            #     self.network.save_model(self.network, self.network.optimizer, epoch, self.iterations, self.options, self.run_start)

            self.iterations += 1

            # Limit iterations per epoch
            if self.options.iterations > 0 and (batch_num + 1) % self.options.iterations == 0:
                break

        del loss, losses_dict
