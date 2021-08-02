import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from configs.options import Options
from utils.utils import get_progress
from utils.models import init_seed_state
from loggings.logger import Logger
from ..testing.fid import FrechetInceptionDistance
from ..testing.ssim import calculate_ssim
from ..dataset.dataset import VoxCelebDataset
from ..dataset.transforms import Resize, GrayScale, ToTensor, Normalize
from ..models.network import Network

from utils.transforms import denormalize
import numpy as np
import cv2

class Tester():
    def __init__(self, logger: Logger, options: Options, network: Network):
        self.logger = logger
        self.options = options
        self.network = network
        self.tag_prefix = self.options.tag_prefix

        torch.backends.cudnn.benchmark = True
        init_seed_state(self.options)

        self.data_loader_test = self._get_data_loader(train_format=True)
        self.network.eval()


    def _get_data_loader(self, train_format):
        transforms_list = [
            Resize(self.options.image_size, train_format),
            GrayScale(train_format) if self.options.channels <= 1 else None,
            ToTensor(self.options.channels, self.options.device, train_format),
            Normalize(self.options.normalize[0], self.options.normalize[1], train_format)
        ]
        transforms_list = [t for t in transforms_list if t is not None]

        dataset_test = VoxCelebDataset(
            self.options.dataset_test,
            self.options.csv_test,
            self.options.image_size,
            self.options.channels,
            self.options.landmark_type,
            transform=transforms.Compose(transforms_list),
            train_format=train_format
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

        fid = FrechetInceptionDistance(self.options, device=self.options.device, data_loader_length=len(self.data_loader_test), batch_size=self.options.batch_size_test)
        iterations = epoch * len(self.data_loader_test) if while_train else 0

        for batch_num, batch in enumerate(self.data_loader_test):
            batch_start = datetime.now()

            images_real = batch['image2']
            images_fake, loss_G, losses_G_dict, loss_D, losses_D_dict = self.network(batch['image1'], batch['landmark2'], batch, calc_loss=True)

            # Calculate FID
            fid.calculate_activations(images_real, images_fake, batch_num)
            # Calculate SSIM
            ssim_val = calculate_ssim(images_fake, images_real, normalize=self.options.normalize)

            batch_end = datetime.now()

            # LOG PROGRESS
            if (batch_num + 1) % 1 == 0 or batch_num == 0:
                progress = get_progress(batch_num, len(self.data_loader_test))
                message = f'{progress} | Time: {batch_end - batch_start}'
                if while_train:
                    message = f'Epoch {epoch + 1}: {message}'
                self.logger.log_info(
                    f'{message} | '
                    f'SSIM = {ssim_val.mean().item():.4f} | '
                    f'Loss_G = {loss_G:.4f} Loss_D = {loss_D:.4f}'
                )
                self.logger.log_scalar('SSIM Validation', ssim_val.mean().item(), iterations)
                self.logger.log_scalars(losses_G_dict, iterations, tag_prefix=self.tag_prefix)
                self.logger.log_scalars(losses_D_dict, iterations, tag_prefix=self.tag_prefix)
                del ssim_val, loss_G, losses_G_dict, loss_D, losses_D_dict

            # LOG GENERATED IMAGES
            images_source = batch['image1'].detach().clone()
            landmarks_target = batch['landmark2'].detach().clone()
            images = torch.cat((images_source, landmarks_target, images_real, images_fake), dim=0)
            self.logger.save_image(self.options.gen_test_dir, f'0_last_result', images, nrow=self.options.batch_size_test)

            if not while_train or (batch_num + 1) % (self.options.log_freq_test) == 0:
                self.logger.save_image(self.options.gen_test_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=iterations, nrow=self.options.batch_size_test)
                self.logger.log_image('Test/Generated', images, iterations, nrow=self.options.batch_size_test)
                del images_real, images_fake, images, images_source, landmarks_target
            else:
                del images_real, images_fake, images, images_source, landmarks_target
            
            iterations += 1
        
        fid_val = fid.calculate_fid()
        self.logger.log_info(f'FID = {fid_val:.4f}')
        self.logger.log_scalar('FID Validation', fid_val, epoch)

        run_end = datetime.now()
        self.logger.log_info(f'Testing finished in {run_end - run_start}.')

        return fid_val

    def generate(self, gen_test_dir, epoch=None):
        run_start = datetime.now()

        while_train = epoch is not None

        self.logger.log_info('===== TESTING =====')
        self.logger.log_info(f'Running on {self.options.device.upper()}.')
        self.logger.log_info(f'Batches/Iterations: {len(self.data_loader_test)} Batch Size: {self.options.batch_size_test}')

        iterations = epoch * len(self.data_loader_test) if while_train else 0

        for batch_num, batch in enumerate(self.data_loader_test):
            batch_start = datetime.now()

            images_real = batch['image2']
            images_fake = self.network(batch['image1'], batch['landmark2'])

            batch_end = datetime.now()

            # LOG PROGRESS
            if (batch_num + 1) % 1 == 0 or batch_num == 0:
                progress = get_progress(batch_num, len(self.data_loader_test))
                message = f'[{progress}] | Time: {batch_end - batch_start} | Experiment: {gen_test_dir}'
                if while_train:
                    message = f'Epoch {epoch + 1}: {message}'
                self.logger.log_info(message)

            # LOG GENERATED IMAGES
            images_source = batch['image1'].detach().clone()
            landmarks_target = batch['landmark2'].detach().clone()
            images = torch.cat((images_source, landmarks_target, images_real, images_fake), dim=0)

            if not while_train or (batch_num + 1) % self.options.log_freq_test == 0:
                self.logger.save_image(gen_test_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=iterations, nrow=self.options.batch_size_test)
                del images_real, images_fake, images, images_source, landmarks_target
            else:
                del images_real, images_fake, images, images_source, landmarks_target
            
            iterations += 1

        run_end = datetime.now()
        self.logger.log_info(f'Testing finished in {run_end - run_start}.')


    def test_batch(self):
        blank = torch.ones((self.options.channels, self.options.image_size, self.options.image_size)).to(self.options.device)
        for batch_num, batch in enumerate(self.data_loader_test):

            img_source = batch['image2'][self.options.batch_size_test // 2 : self.options.batch_size_test]
            lm_source = batch['landmark2'][self.options.batch_size_test // 2 : self.options.batch_size_test]
            img_target = batch['image1'][0:self.options.batch_size_test // 2]

            targets = img_target[0:self.options.batch_size_test // 2]
            for i in range(self.options.batch_size_test // 2):
                t = img_target[i]
                if i == 0:
                    targets = t
                else:
                    targets = torch.cat((targets, t), dim=2)
            targets = torch.cat((blank, targets), dim=2)

            sources = img_source[self.options.batch_size_test // 2 : self.options.batch_size_test]
            for i in range(self.options.batch_size_test // 2):
                s = img_source[i]
                if i == 0:
                    sources = s
                else:
                    sources = torch.cat((sources, s), dim=1)

            gen_total = None
            for r in range(self.options.batch_size_test // 2):
                gen_row = None
                for c in range(self.options.batch_size_test // 2):
                    lm_src = lm_source[r].unsqueeze(0)
                    trg = img_target[c].unsqueeze(0)
                    gen = self.network(trg, lm_src).squeeze(0)
                    if c == 0:
                        gen_row = gen
                    else:
                        gen_row = torch.cat((gen_row, gen), dim=2)
                if r==0:
                    gen_total = gen_row
                else:
                    gen_total = torch.cat((gen_total, gen_row), dim=1)

            src_gen_cat = torch.cat((sources, gen_total), dim=2)
            output = torch.cat((targets, src_gen_cat), dim=1)

            output = denormalize(output, self.options.normalize[0], self.options.normalize[1])
            output = output.cpu().numpy().transpose(1, 2, 0) * 255.0
            output = output.clip(0.0, 255.0).astype(np.uint8)

            cv2.imwrite(f'./results_testing/{batch_num}_output.png', output)
