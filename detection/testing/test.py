import os
from utils.transforms import denormalize, normalize
from utils.models import init_seed_state
from utils.preprocess import detect_crop_face
import torch
import cv2
import numpy as np

from face_alignment import FaceAlignment, LandmarksType
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

from configs.options import Options
from loggings.logger import Logger
from loggings.utils import plot_confusion_matrix, plot_roc_curve, save_cm_roc, plot_prc_curve, save_prc
from utils.utils import add_losses, avg_losses, get_progress, init_class_losses, init_feature_losses
from ..dataset.dataset import FaceForensicsDataset, get_pair_classification, get_pair_feature
from ..dataset.transforms import Resize, GrayScale, ToTensor, Normalize
from ..models.network import Network

class Tester():
    def __init__(self, logger: Logger, options: Options, network: Network):
        self.logger = logger
        self.options = options
        self.network = network
        self.tag_prefix = self.options.tag_prefix
        self.pos_label = 0

        torch.backends.cudnn.benchmark = True
        init_seed_state(self.options, model_name='SiameseResNet')

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
            20, # self.options.max_frames, # TODO: 14b_experiment, 14a_experiment 17a_experiment fix hard coding
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


    def test_classification(self, epoch=None, inf=False):
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
                loss, losses_dict, target, output, mask, image = self.network.forward_classification(batch, backward=False)
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
                    _, _, m, _ = get_pair_classification(batch)
                    images = torch.cat((mask.detach().clone(), m.detach().clone()), dim=0)
                    self.logger.save_image(self.options.gen_test_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=iterations, nrow=batch_size)
                    self.logger.log_image('Test_Mask_Class', images, iterations, nrow=batch_size)

            iterations += 1

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

        self.logger.log_scalar('F_Beta', f_beta, epoch, tag_prefix=self.tag_prefix)

        # Precision
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0.0 
        self.logger.log_scalar('Precision', precision, epoch, tag_prefix=self.tag_prefix)

        # Recall
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0.0
        self.logger.log_scalar('Recall', recall, epoch, tag_prefix=self.tag_prefix)
        
        # LOG EPOCH LOSS
        epoch_loss = avg_losses(epoch_loss, iterations * batch_size)
        epoch_accuracy = epoch_correct / epoch_total
        self.logger.log_scalars(epoch_loss, epoch, tag_prefix=self.tag_prefix)
        self.logger.log_scalar('Accuracy', epoch_accuracy, epoch, tag_prefix=self.tag_prefix)

        # LOG PRC
        total_target = total_target.detach().cpu().numpy()
        total_prediction = total_prediction.detach().cpu().numpy()
        precision_, recall_, prc_thresholds = precision_recall_curve(total_target, total_prediction, pos_label=self.pos_label)

        f_beta_ = (1+beta**2) * ((precision_ * recall_) / ((beta**2 * precision_) + recall_))
        prc_optimal_idx = np.argmax(f_beta_)
        prc_threshold = prc_thresholds[prc_optimal_idx]
        prc_auc = average_precision_score(total_target, total_prediction, pos_label=self.pos_label)
        save_prc(self.options.log_dir, epoch, precision_, recall_, prc_thresholds, prc_threshold, prc_auc)
        prc_curve_path = 'inf_prc_curve' if inf is True else 'prc_curve'
        if not os.path.isdir(os.path.join(self.options.log_dir, prc_curve_path)):
            os.makedirs(os.path.join(self.options.log_dir, prc_curve_path))
        plot_prc_curve(os.path.join(self.options.log_dir, prc_curve_path, f'prc_e_{epoch}.pdf'), precision_, recall_, prc_threshold, prc_auc, prc_optimal_idx)
        self.logger.log_scalar('PRC_AUC', prc_auc, epoch, tag_prefix=self.tag_prefix)


        # LOG AUC
        fpr, tpr, thresholds = roc_curve(total_target, total_prediction, pos_label=self.pos_label)
        roc_auc = auc(fpr, tpr)
        self.logger.log_scalar('AUC', roc_auc, epoch, tag_prefix=self.tag_prefix)

        self.logger.log_info(
            f'End of Epoch {epoch + 1}: | '
            f'Accuracy = {epoch_accuracy:.4f} | AUC = {roc_auc:.4f}'
        )
        self.logger.log_infos(epoch_loss)
        save_cm_roc(self.options.log_dir, epoch, confusion_matrix, fpr, tpr, thresholds, pos_label=self.pos_label)
        cm_roc_path = 'inf_cm_roc' if inf is True else 'cm_roc'
        if not os.path.isdir(os.path.join(self.options.log_dir, cm_roc_path)):
            os.makedirs(os.path.join(self.options.log_dir, cm_roc_path))
        plot_confusion_matrix(os.path.join(self.options.log_dir, cm_roc_path, f'cm_e_{epoch}.pdf'), confusion_matrix.detach().cpu().numpy())
        plot_roc_curve(os.path.join(self.options.log_dir, cm_roc_path, f'roc_e_{epoch}.pdf'), fpr, tpr, thresholds, pos_label=self.pos_label)

        self.logger.log_info(f'Precision = {precision:.4f} | Recall = {recall:.4f} | F_Beta = {f_beta:.4f} | PRC_AUC = {prc_auc:.4f}')

        run_end = datetime.now()
        self.logger.log_info(f'Testing finished in {run_end - run_start}.')

        # return epoch_accuracy
        # # return roc_auc
        # # return f_beta
        return loss


    def test_batch(self):
        self.data_loader_test = self._get_data_loader(batch_size=self.options.batch_size_test // 2)
        tp = [] # 0,0
        fn = [] # 0,1
        fp = [] # 1,0
        tn = [] # 1,1
        tp_tp = []
        fn_tp = []
        fp_tp = []
        tn_tp = []

        for batch_num, batch in enumerate(self.data_loader_test):
            with torch.no_grad():
                _, _, target, output, _, image = self.network.forward_classification(batch, backward=False)

            prediction, _ = torch.max((output > self.options.threshold).float()*1, 1)
            prediction_prob, _ = torch.max(output, 1)

            if batch_num <= 0:
                total_target = target.view(-1)
                total_prediction = prediction_prob.view(-1)
            else:
                total_target = torch.cat((total_target, target.view(-1)))
                total_prediction = torch.cat((total_prediction, prediction_prob.view(-1)))#

            for i, (t, p) in enumerate(zip(target.view(-1), prediction.view(-1))):
                t = t.cpu().numpy()
                p = p.cpu().numpy()

                if t == 0 and p == 0:
                    tp.append(image[i])
                    tp_tp.append([t, p])
                if t == 0 and p == 1:
                    fn.append(image[i])
                    fn_tp.append([t, p])
                if t == 1 and p == 0:
                    fp.append(image[i])
                    fp_tp.append([t, p])
                if t == 1 and p == 1:
                    tn.append(image[i])
                    tn_tp.append([t, p])

        # Save images
        cols = 8
        self.save_image_rows(np.array(tp), trg_pred=np.array(tp_tp), n_cols=cols, path='./detection_results/tp')
        self.save_image_rows(np.array(fn), trg_pred=np.array(fn_tp), n_cols=cols, path='./detection_results/fn')
        self.save_image_rows(np.array(fp), trg_pred=np.array(fp_tp), n_cols=cols, path='./detection_results/fp')
        self.save_image_rows(np.array(tn), trg_pred=np.array(tn_tp), n_cols=cols, path='./detection_results/tn')


    def save_image_rows(self, images, trg_pred, n_cols, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        len_images = len(images)
        color_fake = (0, 0, 255)
        color_real = (0, 255, 0)
        fa = FaceAlignment(LandmarksType._2D, device=self.options.device)

        # Shuffle
        idx = torch.randperm(images.shape[0])
        images = images[idx]
        trg_pred = trg_pred[idx]

        for i in range(0, len_images, n_cols):
            image_row = None

            for j in range(i, i+n_cols):
                st = datetime.now()
                if i+n_cols >= len_images:
                    break

                img_tmp = images[j]
                img_tmp = img_tmp.cpu().numpy().transpose(1, 2, 0) * 255.0
                img_tmp = img_tmp.clip(0.0, 255.0).astype(np.uint8)
                img_tmp, rmin, rmax, cmin, cmax = detect_crop_face(self.options.image_size, img_tmp, padding=self.options.padding, face_alignment=fa, crop=False)
                rmin = np.clip(rmin, 0, img_tmp.shape[0])
                rmax = np.clip(rmax, 0, img_tmp.shape[0])
                cmin = np.clip(cmin, 0, img_tmp.shape[1])
                cmax = np.clip(cmax, 0, img_tmp.shape[1])

                # cv2.rectangle(img_tmp, (0, 0), (img_tmp.shape[1], img_tmp.shape[0]), color=color_fake if not trg_pred[j, 0] else color_real, thickness=2)
                cv2.rectangle(img_tmp, (cmin, rmin), (cmax, rmax), color=color_fake if not trg_pred[j, 1] else color_real, thickness=2)

                img_tmp = img_tmp.transpose(2, 0, 1)
                images[j] = torch.from_numpy(img_tmp).to(self.options.device)

                # Init row image
                if j == i:
                    image_row = images[j]
                else:
                    image_row = torch.cat((images[j], image_row), dim=2)

            if image_row is None:
                break

            image_row = image_row.cpu().numpy().transpose(1, 2, 0)
            image_row = image_row.clip(0.0, 255.0).astype(np.uint8)

            cv2.imwrite(os.path.join(path, f'{i}.png'), image_row)
            et = datetime.now()
            self.logger.log_info(f'{et - st}')
