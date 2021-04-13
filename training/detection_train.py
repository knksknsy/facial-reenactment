import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from configs.train_options import TrainOptions
from testing.detection_test import TesterDetection
from dataset.dataset import FaceForensicsDataset, get_pair_classification, get_pair_feature
from dataset.faceforensics_transforms import Resize, GrayScale, RandomHorizontalFlip, RandomRotate, ToTensor, Normalize
from models.detection_network import NetworkDetection
from models.utils import lr_linear_schedule, init_seed_state
from loggings.logger import Logger
from utils import get_progress


class TrainerDetection():
    def __init__(self, logger: Logger, options: TrainOptions):
        self.logger = logger
        self.options = options
        self.training = True

        torch.backends.cudnn.benchmark = True
        init_seed_state(self.options)

        self.network = NetworkDetection(self.logger, self.options, model_path=None)


    def _get_data_loader(self, batch_size: int):
        transforms_list = [
            Resize(self.options.image_size, self.options.mask_size),
            GrayScale() if self.options.channels <= 1 else None,
            RandomHorizontalFlip() if self.options.horizontal_flip else None,
            RandomRotate(self.options.rotation_angle) if self.options.rotation_angle > 0 else None,
            ToTensor(self.options.channels, self.options.device),
            Normalize(self.options.normalize[0], self.options.normalize[1])
        ]
        transforms_list = [t for t in transforms_list if t is not None]

        dataset_train = FaceForensicsDataset(
            self.options.max_frames,
            self.options.dataset_train,
            self.options.csv_train,
            self.options.image_size,
            self.options.channels,
            transforms.Compose(transforms_list)
        )

        data_loader_train = DataLoader(
            dataset_train,
            batch_size,
            self.options.shuffle,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory,
            drop_last=True
        )
        return data_loader_train


    def start(self):
        self.run_start = datetime.now()

        self.iterations = self.network.continue_iteration

        self.logger.log_info('===== TRAINING =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')
        self.logger.log_info(f'----- OPTIONS -----')
        self.logger.log_info(self.options)
        self.logger.log_info(f'Epochs: {self.options.epochs} Batch Size: {self.options.batch_size}')
        self.logger.log_info(f'Learning Rate = {self.network.optimizer.param_groups[0]["lr"]}')

        for epoch in range(self.network.continue_epoch, self.options.epochs):
            epoch_start = datetime.now()

            self._train(epoch)

            if self.options.test and epoch >= self.options.epochs_feature:
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

        # TODO debugging: change to lt
        # Train feature extractor for epochs_feature epochs
        if epoch > self.options.epochs_feature:
            self._train_feature(self.network.continue_epoch)
        # Train classification for remaining epochs (epochs - epochs_feature)
        else:
            self._train_classification(self.network.continue_epoch)

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
        self.network.save_model(
            model=self.network.siamese_net,
            optimizer=self.network.optimizer,
            scheduler=self.network.scheduler,
            epoch=epoch, iteration=self.iterations,
            options=self.options, time_for_name=self.run_start
        )


    def _train_feature(self, epoch: int):
        self.data_loader_train = self._get_data_loader(batch_size=self.options.batch_size)
        loss_real, loss_fake = 0, 0

        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()
            real_pair = (batch_num + 1) % 2 == 1
            x1, x2, target = get_pair_feature(batch, real_pair=real_pair, device=self.options.device)
            loss = self.network.forward_feature(x1, x2, target)
            loss_real = loss if real_pair else loss_real
            loss_fake = loss if not real_pair else loss_fake
            batch_end = datetime.now()

            # LOG LOSS
            if (batch_num + 1) % self.options.log_freq == 0:
                progress = get_progress(batch_num, len(self.data_loader_train), limit=self.options.iterations if self.options.iterations > 0 else None)
                self.logger.log_info(
                    f'Epoch {epoch + 1}: {progress} | '
                    f'Time: {batch_end - batch_start} | '
                    f'Loss (Contrastive): Real = {loss_real:.8f} Fake = {loss_fake:.8f}'
                )
            if real_pair:
                self.logger.log_scalar('Loss_Contrastive_Real', loss, self.iterations)
            else:
                self.logger.log_scalar('Loss_Contrastive_Fake', loss, self.iterations)
            self.logger.log_scalar('Loss_Contrastive', loss, self.iterations)

            # Limit iterations per epoch
            self.iterations += 1
            if self.options.iterations > 0 and (batch_num + 1) % self.options.iterations == 0:
                break
        del loss, loss_real, loss_fake


    def _train_classification(self, epoch: int):
        self.data_loader_train = self._get_data_loader(batch_size=self.options.batch_size//2)
        correct = 0
        total = 0
        confusion_matrix = torch.zeros(2, 2).to(self.options.device)

        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()

            x, target = get_pair_classification(batch)
            loss, output = self.network.forward_classification(x, target)
            
            # ACCURACY
            _, prediction = torch.max(torch.round(output), 1)
            total += target.shape[0]
            correct += (prediction == target.squeeze()).sum().item()
            accuracy = correct / total

            # TODO
            # CONFUSION MATRIX
            for t, p in zip(target.view(-1), prediction.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            # self.logger.log_info(confusion_matrix)
            # Get Per-class accuracy
            # confusion_matrix.diag()/confusion_matrix.sum(1)

            batch_end = datetime.now()

            # LOG LOSS
            if (batch_num + 1) % self.options.log_freq == 0:
                progress = get_progress(batch_num, len(self.data_loader_train), limit=self.options.iterations if self.options.iterations > 0 else None)
                self.logger.log_info(
                    f'Epoch {epoch + 1}: {progress} | '
                    f'Time: {batch_end - batch_start} | '
                    f'Loss (BCE) = {loss:.4f} | '
                    f'Accuracy = {accuracy:.2f}'
                )
            self.logger.log_scalar('Loss_BCE', loss, self.iterations)
            self.logger.log_scalar('Train_Accuracy', accuracy, self.iterations)

            # Limit iterations per epoch
            self.iterations += 1
            if self.options.iterations > 0 and (batch_num + 1) % self.options.iterations == 0:
                break

        # LOG CONFUSION MATRIX
        
        del loss
