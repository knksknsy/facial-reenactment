import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from configs.options import Options
from dataset.dataset import FaceForensicsDataset
from dataset.faceforensics_transforms import Resize, GrayScale, ToTensor, Normalize
from models.detection_network import NetworkDetection
from loggings.logger import Logger

class TesterDetection():
    def __init__(self, logger: Logger, options: Options, network: NetworkDetection):
        self.logger = logger
        self.options = options
        self.network = network

        self.data_loader_test = self._get_data_loader()
        self.network.eval()


    def _get_data_loader(self, train_format):
        transforms_list = [
            Resize(self.options.image_size, train_format),
            GrayScale(train_format) if self.options.channels <= 1 else None,
            ToTensor(self.options.channels, self.options.device, train_format),
            Normalize(self.options.normalize[0], self.options.normalize[1], train_format)
        ]
        transforms_list = [t for t in transforms_list if t is not None]

        dataset_test = FaceForensicsDataset(
            self.options.dataset_test,
            self.options.csv_test,
            self.options.image_size,
            self.options.channels,
            transform=transforms.Compose(transforms_list),
        )

        data_loader_test = DataLoader(
            dataset_test,
            self.options.batch_size_test,
            self.options.shuffle_test,
            num_workers=self.options.num_workers_test,
            pin_memory=self.options.pin_memory,
            drop_last=True
        )

        return data_loader_test


    def test(self, epoch=None):
        run_start = datetime.now()

        while_train = epoch is not None

        self.logger.log_info('===== TESTING =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')
        self.logger.log_info(f'Batches/Iterations: {len(self.data_loader_test)} Batch Size: {self.options.batch_size_test}')

        iterations = epoch * len(self.data_loader_test) if while_train else 0

        for batch_num, batch in enumerate(self.data_loader_test):
            batch_start = datetime.now()

            images_real = batch['images_real']
            images_fake = batch['images_fake']
            labels_real = batch['labels_real']
            labels_fake = batch['labels_fake']
            preds, features, loss, losses_dict = self.network(images_real, images_fake, labels_real, labels_fake, calc_loss=True)

            # Calculate Metrics

            batch_end = datetime.now()

            # LOG PROGRESS
            if (batch_num + 1) % 1 == 0 or batch_num == 0:
                message = f'[{batch_num + 1}/{len(self.data_loader_test)}] | Time: {batch_end - batch_start}'
                if while_train:
                    message = f'Epoch {epoch + 1}: {message}'
                self.logger.log_info(f'{message} | '
                                    f'METRIC = {0:.4f} | '
                                    f'Loss = {loss:.4f}')
                self.logger.log_scalar('METRIC Validation', 0, iterations)
                self.logger.log_scalars(losses_dict, iterations, tag_prefix='Test')
                del loss, losses_dict

            # LOG LATEST FEATURES
            images_real = batch['images_real'].detach().clone()
            images_fake = batch['images_fake'].detach().clone()
            images_features = features.detach().clone()
            images = torch.cat((images_real, images_fake, images_features), dim=0)
            self.logger.save_image(self.options.gen_dir, f'0_last_result', images, nrow=self.options.batch_size_test)

            if not while_train or (batch_num + 1) % (self.options.log_freq_test) == 0:
                self.logger.save_image(self.options.gen_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=self.iterations, nrow=self.options.batch_size_test)
                self.logger.log_image('Train/Features', images, self.iterations, nrow=self.options.batch_size_test)
                del images_real, images_fake, images
            else:
                del images_real, images_fake, images
            
            iterations += 1
        
        run_end = datetime.now()
        self.logger.log_info(f'Testing finished in {run_end - run_start}.')
