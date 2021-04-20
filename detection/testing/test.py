import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import date, datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import codecs
import json
import numpy as np

from configs.options import Options
from loggings.logger import Logger
from utils.utils import add_losses, avg_losses, get_progress, init_class_losses, init_feature_losses
from ..dataset.dataset import FaceForensicsDataset, get_pair_classification, get_pair_feature
from ..dataset.transforms import Resize, GrayScale, ToTensor, Normalize
from ..models.network import Network

class Tester():
    def __init__(self, logger: Logger, options: Options, network: Network):
        self.logger = logger
        self.options = options
        self.network = network

        self.network.eval()


    def _get_data_loader(self, batch_size: int = None):
        transforms_list = [
            Resize(self.options.image_size, self.options.mask_size),
            GrayScale() if self.options.channels <= 1 else None,
            ToTensor(self.options.channels, self.options.device),
            Normalize(self.options.normalize[0], self.options.normalize[1])
        ]
        transforms_list = [t for t in transforms_list if t is not None]

        dataset_test = FaceForensicsDataset(
            20, #self.options.max_frames, # TODO: 14b_experiment, 17a_experiment fix hard coding
            self.options.dataset_test,
            self.options.csv_test,
            self.options.image_size,
            self.options.channels,
            transform=transforms.Compose(transforms_list),
        )

        data_loader_test = DataLoader(
            dataset_test,
            self.options.batch_size_test if batch_size is None else batch_size,
            self.options.shuffle_test,
            num_workers=self.options.num_workers_test,
            pin_memory=self.options.pin_memory,
            drop_last=True
        )

        return data_loader_test


    def test_feature(self, epoch=None):
        run_start = datetime.now()

        while_train = epoch is not None

        self.logger.log_info('===== TESTING =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')

        self.data_loader_test = self._get_data_loader()
        batch_size = self.options.batch_size_test
        iterations = epoch * len(self.data_loader_test) if while_train else 0

        run_loss = init_feature_losses()
        epoch_loss = init_feature_losses()

        for batch_num, batch in enumerate(self.data_loader_test):
            batch_start = datetime.now()
            real_pair = (batch_num + 1) % 2 == 1
            with torch.no_grad():
                loss, losses_dict, mask1, mask2 = self.network.forward_feature(batch, real_pair=real_pair, backward=False)
            batch_end = datetime.now()

            # LOSS
            run_loss = add_losses(run_loss, losses_dict)
            epoch_loss = add_losses(epoch_loss, losses_dict, batch_size)

            # LOG RUN LOSS
            if (batch_num + 1) % self.options.log_freq_test == 0:
                progress = get_progress(batch_num, len(self.data_loader_test))
                message = f'[{progress}] | Time: {batch_end - batch_start}'
                if while_train:
                    message = f'Epoch {epoch + 1}: {message}'
                self.logger.log_info(message)
                run_loss = avg_losses(run_loss, self.options.log_freq_test)
                self.logger.log_infos(run_loss)
                run_loss = init_feature_losses()
                # LOG MASK
                if self.options.l_mask > 0:
                    _, _, _, m1, m2, = get_pair_feature(batch, real_pair=real_pair, device=self.options.device)
                    images = torch.cat((mask1.detach().clone(), m1.detach().clone(), mask2.detach().clone(), m2.detach().clone()), dim=0)
                    self.logger.save_image(self.options.gen_test_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=iterations, nrow=batch_size)
                    self.logger.log_image('Test_Mask_Feature', images, iterations, nrow=batch_size)

            iterations += 1

        # LOG EPOCH LOSS
        epoch_loss = avg_losses(epoch_loss, iterations * batch_size)
        self.logger.log_info(f'End of Epoch {epoch + 1}')
        self.logger.log_infos(epoch_loss)
        self.logger.log_scalars(epoch_loss, epoch, tag_prefix='Test')
        
        run_end = datetime.now()
        self.logger.log_info(f'Testing finished in {run_end - run_start}.')


    def test_classification(self, epoch=None):
        run_start = datetime.now()

        while_train = epoch is not None

        self.logger.log_info('===== TESTING =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')

        self.data_loader_test = self._get_data_loader(batch_size=self.options.batch_size_test // 2)
        batch_size = self.options.batch_size_test
        iterations = epoch * len(self.data_loader_test) if while_train else 0

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

        for batch_num, batch in enumerate(self.data_loader_test):
            batch_start = datetime.now()
            with torch.no_grad():
                loss, losses_dict, target, output, mask = self.network.forward_classification(batch, backward=False)
            batch_end = datetime.now()

            # LOSS
            run_loss = add_losses(run_loss, losses_dict)
            epoch_loss = add_losses(epoch_loss, losses_dict)
            # ACCURACY
            prediction, _ = torch.max(torch.round(output), 1)
            prediction_prob, _ = torch.max(output, 1)
            # # TODO: Logits
            # prediction, _ = torch.max((output > 0).float() * 1, 1)
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
            if (batch_num + 1) % self.options.log_freq_test == 0:
                progress = get_progress(batch_num, len(self.data_loader_test))
                message = f'[{progress}] | Time: {batch_end - batch_start}'
                if while_train:
                    message = f'Epoch {epoch + 1}: {message}'
                self.logger.log_info(f'{message} | Auccuracy = {(run_correct / run_total):.4f}')
                run_loss = avg_losses(run_loss, self.options.log_freq_test)
                self.logger.log_infos(run_loss)
                run_loss = init_class_losses()
                run_total, run_correct = 0, 0
                # LOG MASK
                if self.options.l_mask > 0:
                    _, _, m, = get_pair_classification(batch)
                    images = torch.cat((mask.detach().clone(), m.detach().clone()), dim=0)
                    self.logger.save_image(self.options.gen_test_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=iterations, nrow=batch_size)
                    self.logger.log_image('Test_Mask_Class', images, iterations, nrow=batch_size)

            iterations += 1
        
        # LOG EPOCH LOSS
        epoch_loss = avg_losses(epoch_loss, iterations * batch_size)
        epoch_accuracy = epoch_correct / epoch_total
        self.logger.log_scalars(epoch_loss, epoch, tag_prefix='Test')
        self.logger.log_scalar('Test_Accuracy', epoch_accuracy, epoch)

        # LOG AUC
        total_target = total_target.detach().cpu().numpy()
        total_prediction = total_prediction.detach().cpu().numpy()
        fpr, tpr, _ = roc_curve(total_target, total_prediction)
        roc_auc = auc(fpr, tpr)
        self.logger.log_scalar('Test_AUC', roc_auc, epoch)

        self.logger.log_info(
            f'End of Epoch {epoch + 1}: | '
            f'Accuracy = {epoch_accuracy:.4f} | AUC = {roc_auc:.4f}'
        )
        self.logger.log_infos(epoch_loss)
        self.save_cm_roc(epoch, confusion_matrix, fpr, tpr, roc_auc)
        
        run_end = datetime.now()
        self.logger.log_info(f'Testing finished in {run_end - run_start}.')

        return epoch_accuracy


    def save_cm_roc(self, epoch: int, cm, fpr, tpr, roc_auc):
        path = os.path.join(self.options.log_dir, 'cm_roc')
        if not os.path.isdir(path):
            os.makedirs(path)

        json_dict = dict()
        json_dict['cm'] = cm.tolist()
        json_dict['roc'] = dict()
        json_dict['roc']['fpr'] = fpr.tolist()
        json_dict['roc']['tpr'] = tpr.tolist()
        json_dict['roc']['roc_auc'] = roc_auc

        path = os.path.join(path, f'cm_roc_e_{epoch}.json')
        json.dump(json_dict, codecs.open(path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=4)
