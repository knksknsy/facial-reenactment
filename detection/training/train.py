import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import roc_curve, auc

from configs.train_options import TrainOptions
from utils.models import lr_linear_schedule, init_seed_state
from loggings.logger import Logger
from utils.utils import get_progress

from ..testing.test import Tester
from ..dataset.dataset import FaceForensicsDataset, get_pair_classification, get_pair_feature
from ..dataset.transforms import Resize, GrayScale, RandomHorizontalFlip, RandomRotate, ToTensor, Normalize
from ..models.network import Network


class Trainer():
    def __init__(self, logger: Logger, options: TrainOptions):
        self.logger = logger
        self.options = options
        self.training = True

        torch.backends.cudnn.benchmark = True
        init_seed_state(self.options)

        self.network = Network(self.logger, self.options, model_path=None)


    def _get_data_loader(self, batch_size: int = None):
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
            self.options.batch_size if batch_size is None else batch_size,
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

            if self.options.test:
                test = Tester(self.logger, self.options, self.network)
                # Test feature extraction
                if epoch >= self.options.epochs_feature:
                    accuracy = test.test_classification(epoch)
                    # Decrease LR if accuracy stagnates
                    if 'lr_plateau_decay' in self.options.config['train']['optimizer']:
                        self.network.scheduler.step(accuracy)

                # Test classification
                if epoch < self.options.epochs_feature:
                    test.test_feature(epoch)

            # Create new tensorboard event for each epoch
            self.logger.init_writer(filename_suffix=f'.{str(epoch+1)}')

            epoch_end = datetime.now()
            self.logger.log_info(f'Epoch {epoch + 1} finished in {epoch_end - epoch_start}.')

        self.run_end = datetime.now()
        self.logger.log_info(f'Training finished in {self.run_end - self.run_start}.')


    def _train(self, epoch):
        self.network.train()

        # Train feature extractor for epochs_feature epochs
        if epoch < self.options.epochs_feature:
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
        self.data_loader_train = self._get_data_loader()
        epoch_loss, epoch_loss_real, epoch_loss_fake = 0.0, 0.0, 0.0
        run_loss, run_loss_real, run_loss_fake = 0.0, 0.0, 0.0

        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()
            real_pair = (batch_num + 1) % 2 == 1
            x1, x2, target = get_pair_feature(batch, real_pair=real_pair, device=self.options.device)
            loss = self.network.forward_feature(x1, x2, target)
            batch_end = datetime.now()

            # LOSS
            run_loss += loss
            epoch_loss += x1.shape[0] * loss

            if real_pair:
                run_loss_real += loss
                epoch_loss_real += x1.shape[0] * loss
            else:
                run_loss_fake += loss
                epoch_loss_fake += x1.shape[0] * loss

            # LOG RUN LOSS
            if (batch_num + 1) % self.options.log_freq == 0:
                progress = get_progress(batch_num, len(self.data_loader_train), limit=self.options.iterations if self.options.iterations > 0 else None)
                self.logger.log_info(
                    f'Epoch {epoch + 1}: {progress} | '
                    f'Time: {batch_end - batch_start} | '
                    f'Loss (Contrastive): '
                    f'Total = {(run_loss / self.options.log_freq):.8f} '
                    f'Real = {(run_loss_real / self.options.log_freq / 2):.8f} '
                    f'Fake = {(run_loss_fake / self.options.log_freq / 2):.8f} '
                )
                run_loss, run_loss_real, run_loss_fake = 0.0, 0.0, 0.0

            # Limit iterations per epoch
            self.iterations += 1
            if self.options.iterations > 0 and (batch_num + 1) % self.options.iterations == 0:
                break

        # LOG EPOCH LOSS
        epoch_loss = epoch_loss / (self.iterations * self.options.batch_size)
        epoch_loss_real = epoch_loss_real / (self.iterations * self.options.batch_size / 2)
        epoch_loss_fake = epoch_loss_fake / (self.iterations * self.options.batch_size / 2)
        self.logger.log_info(f'Epoch {epoch + 1}: Loss (Contrastive): Total = {epoch_loss} Real = {epoch_loss_real} Fake = {epoch_loss_fake}')
        self.logger.log_scalar('Loss_Contrastive', epoch_loss, epoch)
        self.logger.log_scalar('Loss_Contrastive_Real', epoch_loss_real, epoch)
        self.logger.log_scalar('Loss_Contrastive_Fake', epoch_loss_fake, epoch)


    def _train_classification(self, epoch: int):
        self.data_loader_train = self._get_data_loader(batch_size=self.options.batch_size // 2)

        epoch_loss, run_loss = 0.0, 0.0
        epoch_correct, run_correct = 0, 0
        run_total = 0

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        total_target = None
        total_prediction = None

        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()
            x, target = get_pair_classification(batch)
            loss, output = self.network.forward_classification(x, target)
            batch_end = datetime.now()

            # LOSS
            run_loss += loss
            epoch_loss += x.shape[0] * loss
            # ACCURACY
            prediction, _ = torch.max(torch.round(output), 1)
            prediction_prob, _ = torch.max(output, 1)
            run_total += x.shape[0]
            run_correct += (prediction == target.squeeze()).sum().item()
            epoch_correct += (prediction == target.squeeze()).sum().item()
            # ROC + AUC
            if batch_num <= 0:
                total_target = target.view(-1)
                total_prediction = prediction_prob.view(-1)
            else:
                total_target = torch.cat((total_target, target.view(-1)))
                total_prediction = torch.cat((total_prediction, prediction_prob.view(-1)))

            # LOG RUN LOSS
            if (batch_num + 1) % self.options.log_freq == 0:
                progress = get_progress(batch_num, len(self.data_loader_train), limit=self.options.iterations if self.options.iterations > 0 else None)
                self.logger.log_info(
                    f'Epoch {epoch + 1}: {progress} | '
                    f'Time: {batch_end - batch_start} | '
                    f'Loss (BCE) = {(run_loss / self.options.log_freq):.8f} | '
                    f'Accuracy = {(run_correct / run_total):.4f}'
                )
                run_loss, run_total, run_correct = 0.0, 0, 0

            # Limit iterations per epoch
            self.iterations += 1
            if self.options.iterations > 0 and (batch_num + 1) % self.options.iterations == 0:
                break

        # LOG EPOCH LOSS
        epoch_loss = epoch_loss / (self.iterations * self.options.batch_size)
        epoch_accuracy = epoch_correct / (self.iterations * self.options.batch_size)
        self.logger.log_scalar('Loss_BCE', epoch_loss, epoch)
        self.logger.log_scalar('Accuracy', epoch_accuracy, epoch)

        # LOG AUC
        total_target = total_target.detach().cpu().numpy()
        total_prediction = total_prediction.detach().cpu().numpy()
        fpr, tpr, _ = roc_curve(total_target, total_prediction)
        roc_auc = auc(fpr, tpr)
        self.logger.log_scalar('AUC', roc_auc, epoch)

        self.logger.log_info(f'Epoch {epoch + 1}: Loss (BCE) = {epoch_loss:.8f} | Accuracy = {epoch_accuracy:.4f} | AUC = {roc_auc:.4f}')
