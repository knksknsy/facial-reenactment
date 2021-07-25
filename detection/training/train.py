import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score

from configs.train_options import TrainOptions
from utils.models import lr_linear_schedule, init_seed_state
from loggings.logger import Logger
from utils.utils import add_losses, avg_losses, get_progress, init_class_losses, init_feature_losses

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
        init_seed_state(self.options, model_name='SiameseResNet')

        self.network = Network(self.logger, self.options, model_path=None)


    def _get_data_loader(self, dataset_train=None, csv_train=None, max_frames=None, batch_size: int = None):
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
            self.options.max_frames if max_frames is None else max_frames,
            self.options.dataset_train if dataset_train is None else dataset_train,
            self.options.csv_train if csv_train is None else csv_train,
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
                if epoch < self.options.epochs_feature:
                    test.test_feature(epoch)
                # Test classification
                else:
                    metric = test.test_classification(epoch)
                    # Decrease LR if metric stagnates
                    if 'lr_plateau_decay' in self.options.config['train']['optimizer']:
                        self.network.scheduler.step(metric)

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
            self._train_feature(epoch)
        # Train classification for remaining epochs (epochs - epochs_feature)
        else:
            self._train_classification(epoch)

        # Schedule learning rate
        if 'lr_linear_decay' in self.options.config['train']['optimizer']:
            self.network.optimizer.param_groups[0]['lr'] = lr_linear_schedule(
                epoch,
                epoch_start=self.options.epoch_range[0],
                epoch_end=self.options.epoch_range[1],
                lr_base=self.options.lr,
                lr_end=self.options.lr_end
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
        # # 14b_experiment 17a_experiment
        # self.data_loader_train = self._get_data_loader(dataset_train='/home/kaan/datasets/FaceForensics/Preprocessed/dev/', csv_train='./csv/faceforensics_dev.csv', max_frames=20)
        self.data_loader_train = self._get_data_loader()
        batch_size = self.options.batch_size

        run_loss = init_feature_losses()
        epoch_loss = init_feature_losses()

        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()
            real_pair = (batch_num + 1) % 2 == 1
            loss, losses_dict, mask1, mask2 = self.network.forward_feature(batch, real_pair=real_pair)
            batch_end = datetime.now()

            # LOSS
            run_loss = add_losses(run_loss, losses_dict)
            epoch_loss = add_losses(epoch_loss, losses_dict, batch_size)

            # LOG RUN LOSS
            if (batch_num + 1) % self.options.log_freq == 0:
                progress = get_progress(batch_num, len(self.data_loader_train), limit=self.options.iterations if self.options.iterations > 0 else None)
                self.logger.log_info(f'Epoch {epoch + 1}: {progress} | Time: {batch_end - batch_start}')
                run_loss = avg_losses(run_loss, self.options.log_freq)
                self.logger.log_infos(run_loss)
                run_loss = init_feature_losses()
                # LOG MASK
                if self.options.l_mask > 0:
                    _, _, _, m1, m2, = get_pair_feature(batch, real_pair=real_pair, device=self.options.device)
                    images = torch.cat((mask1.detach().clone(), m1.detach().clone(), mask2.detach().clone(), m2.detach().clone()), dim=0)
                    self.logger.save_image(self.options.gen_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=self.iterations, nrow=batch_size)
                    self.logger.log_image('Train_Mask_Feature', images, self.iterations, nrow=batch_size)

            if 'lr_cyclic_decay' in self.options.config['train']['optimizer']:
                self.network.scheduler.step()

            # Limit iterations per epoch
            self.iterations += 1
            if self.options.iterations > 0 and (batch_num + 1) % self.options.iterations == 0:
                break

        # LOG EPOCH LOSS
        epoch_loss = avg_losses(epoch_loss, self.iterations * batch_size)
        self.logger.log_info(f'End of Epoch {epoch + 1}')
        self.logger.log_infos(epoch_loss)
        self.logger.log_scalars(epoch_loss, epoch)


    def _train_classification(self, epoch: int):
        self.data_loader_train = self._get_data_loader(batch_size=self.options.batch_size_class // 2)
        batch_size = self.options.batch_size_class

        run_loss = init_class_losses()
        epoch_loss = init_class_losses()
        epoch_correct, run_correct = 0, 0
        run_total, epoch_total = 0, 0

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        total_target = None
        total_prediction = None
        confusion_matrix = torch.zeros(2, 2).to(self.options.device)

        # batch_size=128/2=64
        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()
            loss, losses_dict, target, output, mask = self.network.forward_classification(batch)
            batch_end = datetime.now()

            # LOSS
            run_loss = add_losses(run_loss, losses_dict)
            epoch_loss = add_losses(epoch_loss, losses_dict)
            # ACCURACY
            prediction, _ = torch.max((output > self.options.threshold).float()*1, 1)
            prediction_prob, _ = torch.max(output, 1)
            # # TODO: Logits
            # prediction_prob, _ = torch.max(torch.sigmoid(output), 1)
            run_total += batch_size
            epoch_total += batch_size
            run_correct += (prediction == target.squeeze()).sum().item()
            epoch_correct += (prediction == target.squeeze()).sum().item()
            # ROC + AUC
            if batch_num <= 0:
                total_target = target.view(-1)
                total_prediction = prediction_prob.view(-1)
            else:
                total_target = torch.cat((total_target, target.view(-1)))
                total_prediction = torch.cat((total_prediction, prediction_prob.view(-1)))

            # CONFUSION MATRIX
            for t, p in zip(target.view(-1), prediction.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            # Get Per-class accuracy
            # confusion_matrix.diag()/confusion_matrix.sum(1)

            # LOG RUN LOSS
            if (batch_num + 1) % self.options.log_freq == 0:
                progress = get_progress(batch_num, len(self.data_loader_train), limit=self.options.iterations if self.options.iterations > 0 else None)
                self.logger.log_info(
                    f'Epoch {epoch + 1}: {progress} | Time: {batch_end - batch_start} | '
                    f'Accuracy = {(run_correct / run_total):.4f}'    
                )
                run_loss = avg_losses(run_loss, self.options.log_freq)
                self.logger.log_infos(run_loss)
                run_loss = init_class_losses()
                run_total, run_correct = 0, 0
                # LOG MASK
                if self.options.l_mask > 0:
                    _, _, m, _ = get_pair_classification(batch)
                    images = torch.cat((mask.detach().clone(), m.detach().clone()), dim=0)
                    self.logger.save_image(self.options.gen_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=self.iterations, nrow=batch_size)
                    self.logger.log_image('Train_Mask_Class', images, self.iterations, nrow=batch_size)

            if 'lr_cyclic_decay' in self.options.config['train']['optimizer']:
                self.network.scheduler.step()

            # Limit iterations per epoch
            self.iterations += 1
            if self.options.iterations > 0 and (batch_num + 1) % self.options.iterations == 0:
                break

        # F1-Score
        tp = confusion_matrix[0,0].item()
        fn = confusion_matrix[0,1].item()
        fp = confusion_matrix[1,0].item()
        tn = confusion_matrix[1,1].item()

        beta = 2 # weigh recall more than precision 
        try:
            f_beta = ((1 + beta**2) * tp) / ((1 + beta**2) * tp + beta**2 * fn + fp)
        except ZeroDivisionError:
            f_beta = 0.0

        self.logger.log_scalar('F_Beta', f_beta, epoch)

        # Precision
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0.0 
        # precision = tp / (tp + fp) if tp > 0 and fp > 0 else 0.0
        self.logger.log_scalar('Precision', precision, epoch)

        # Recall
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0.0
        # recall = tp / (tp + fn) if tp > 0 and fn > 0 else 0.0
        self.logger.log_scalar('Recall', recall, epoch)

        # LOG EPOCH LOSS
        epoch_loss = avg_losses(epoch_loss, self.iterations * batch_size)
        epoch_accuracy = epoch_correct / epoch_total
        self.logger.log_scalars(epoch_loss, epoch)
        self.logger.log_scalar('Accuracy', epoch_accuracy, epoch)

        # LOG PRC
        total_target = total_target.detach().cpu().numpy()
        total_prediction = total_prediction.detach().cpu().numpy()
        prc_auc = average_precision_score(total_target, total_prediction)
        self.logger.log_scalar('PRC_AUC', prc_auc, epoch)

        # LOG AUC
        fpr, tpr, _ = roc_curve(total_target, total_prediction)
        roc_auc = auc(fpr, tpr)
        self.logger.log_scalar('AUC', roc_auc, epoch)

        self.logger.log_info(
            f'End of Epoch {epoch + 1}: | '
            f'Accuracy = {epoch_accuracy:.4f} | AUC = {roc_auc:.4f}'
        )
        self.logger.log_info(f'Precision = {precision:.4f} | Recall = {recall:.4f} | F_Beta = {f_beta:.4f} | PRC_AUC = {prc_auc:.4f}')
        self.logger.log_infos(epoch_loss)
