import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from configs.options import Options
from testing.fid import FrechetInceptionDistance
from testing.ssim import calculate_ssim
from dataset.dataset import VoxCelebDataset
from dataset.transforms import Resize, ToTensor, Normalize
from models.network import Network
from loggings.logger import Logger

class Test():
    def __init__(self, logger: Logger, options: Options, network: Network):
        self.logger = logger
        self.options = options
        self.network = network
        self.data_loader_test = self._get_data_loader(train_format=False)
        self.network.eval()


    def _get_data_loader(self, train_format):
        dataset_test = VoxCelebDataset(
            self.options.dataset_test,
            self.options.csv_test,
            shuffle_frames=False,
            transform=transforms.Compose([
                Resize(self.options.image_size, train_format),
                ToTensor(self.options.device, train_format),
                Normalize(0.5, 0.5, train_format)
            ]),
            train_format=train_format
        )

        data_loader_test = DataLoader(
            dataset_test,
            self.options.batch_size_test,
            self.options.shuffle,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory
        )

        return data_loader_test


    def test(self, epoch=None):
        run_start = datetime.now()

        while_train = epoch is not None

        self.logger.log_info('===== TESTING =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')
        self.logger.log_info(f'Batches/Iterations: {len(self.data_loader_test)} Batch Size: {self.options.batch_size_test}')

        fid = FrechetInceptionDistance(self.options, self.options.device, len(self.data_loader_test))
        iterations = epoch * len(self.data_loader_test) if while_train else 0

        for batch_num, batch in enumerate(self.data_loader_test):
            batch_start = datetime.now()

            images_real = batch['image2']
            images_fake = self.network(batch['image1'], batch['landmark2'])

            # Calculate FID
            fid.calculate_activations(images_real, images_fake, batch_num)
            # Calculate SSIM
            ssim_val = calculate_ssim(images_fake, images_real)

            batch_end = datetime.now()

            # LOG PROGRESS
            if (batch_num + 1) % 1 == 0 or batch_num == 0:
                message = f'[{batch_num + 1}/{len(self.data_loader_test)}] | Time: {batch_end - batch_start}'
                if while_train:
                    message = f'Epoch {epoch + 1}: {message}'
                self.logger.log_info(message)
                self.logger.log_info(f'SSIM = {ssim_val.mean().item():.4f}')
                self.logger.log_scalar('SSIM Validation', ssim_val.mean().item(), iterations)
                del ssim_val

            # LOG GENERATED IMAGES
            images_source = batch['image1'].detach().clone()
            landmarks_target = batch['landmark2'].detach().clone()
            images = torch.cat((images_source, landmarks_target, images_real, images_fake), dim=0)
            self.logger.save_image(self.options.gen_test_dir, f'0_last_result', images, nrow=self.options.batch_size_test)

            if not while_train or (batch_num + 1) % (self.options.log_freq // 10) == 0:
                self.logger.save_image(self.options.gen_test_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=iterations, nrow=self.options.batch_size_test)
                self.logger.log_image('Test/Generated', images, iterations, nrow=self.options.batch_size_test)
                del images_real, images_fake, images, images_source, landmarks_target
            else:
                del images_real, images_fake, images, images_source, landmarks_target
            
            iterations += 1
        
        fid_val = fid.calculate_fid()
        self.logger.log_info(f'FID = {fid_val:.4f}')
        self.logger.log_scalar('FID Validation', fid_val, epoch)
        del fid_val

        run_end = datetime.now()
        self.logger.log_info(f'Testing finished in {run_end - run_start}.')


    def generate(self):
        run_start = datetime.now()

        self.logger.log_info('===== GENERATING FAKE DATASET =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')
        self.logger.log_info(f'Batches/Iterations: {len(self.data_loader_test)} Batch Size: {self.options.batch_size_test}')

        iterations = 0

        for batch_num, batch in enumerate(self.data_loader_test):
            batch_start = datetime.now()

            images_real = batch['image2']
            images_fake = self.network(batch['image1'], batch['landmark2'])

            batch_end = datetime.now()

            # LOG PROGRESS
            if (batch_num + 1) % 1 == 0 or batch_num == 0:
                self.logger.log_info(f'[{batch_num + 1}/{len(self.data_loader_test)}] | Time: {batch_end - batch_start}')

            # LOG GENERATED IMAGES
            self.logger.save_image(self.options.gen_test_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images_fake, epoch=self.network.continue_epoch, iteration=iterations, nrow=self.options.batch_size)
            del images_real, images_fake
            
            iterations += 1

        run_end = datetime.now()
        self.logger.log_info(f'Generating fake dataset finished in {run_end - run_start}.')
