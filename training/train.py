import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

from configs.train_options import TrainOptions
from testing.test import Test
from testing.ssim import calculate_ssim
from testing.fid import FrechetInceptionDistance
from dataset.dataset import VoxCelebDataset
from dataset.transforms import Resize, RandomHorizontalFlip, RandomRotate, ToTensor, Normalize
from models.network import Network
from models.utils import lr_linear_schedule, load_seed_state
from loggings.logger import Logger

class Train():
    def __init__(self, logger: Logger, options: TrainOptions):
        self.logger = logger
        self.options = options
        self.training = True

        torch.backends.cudnn.benchmark = True
        # Set seeds
        if self.options.continue_id is None:
            torch.manual_seed(self.options.seed)
            np.random.seed(self.options.seed)
        # Load seed states
        else:
            numpy_seed_state, torch_seed_state = load_seed_state(self.options)
            torch.set_rng_state(torch_seed_state)
            np.random.set_state(numpy_seed_state)

        self.data_loader_train = self._get_data_loader(train_format=self.training)
        self.network = Network(self.logger, self.options, model_path=None)
        self.fid = FrechetInceptionDistance(self.options, device='cpu', data_loader_length=1)

        # Start training
        self()


    def _get_data_loader(self, train_format):
        transforms_list = [
            Resize(self.options.image_size, train_format),
            RandomHorizontalFlip(train_format) if self.options.horizontal_flip else None,
            RandomRotate(self.options.rotation_angle, train_format) if self.options.rotation_angle > 0 else None,
            ToTensor(self.options.device, train_format),
            Normalize(0.5, 0.5, train_format)
        ]
        compose = [t for t in transforms_list if t is not None]

        dataset_train = VoxCelebDataset(
            self.options.dataset_train,
            self.options.csv_train,
            self.options.shuffle_frames,
            transforms.Compose(compose),
            train_format
        )

        data_loader_train = DataLoader(
            dataset_train,
            self.options.batch_size,
            self.options.shuffle,
            num_workers=self.options.num_workers,
            pin_memory=self.options.pin_memory
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
        self.logger.log_info(f'Learning Rates: Generator = {self.network.optimizer_G.param_groups[0]["lr"]} Discriminator = {self.network.optimizer_D.param_groups[0]["lr"]}')

        for epoch in range(self.network.continue_epoch, self.options.epochs):
            epoch_start =datetime.now()

            self._train(epoch)

            if self.options.test:
                Test(self.logger, self.options, self.network).test(epoch)

            # Create new tensorboard event for each epoch
            self.logger.init_writer()

            epoch_end = datetime.now()
            self.logger.log_info(f'Epoch {epoch + 1} finished in {epoch_end - epoch_start}.')

        self.run_end = datetime.now()
        self.logger.log_info(f'Training finished in {self.run_end - self.run_start}.')


    def _train(self, epoch):
        self.network.train()

        # TRAIN EPOCH
        self._train_epoch(epoch)
        
        # Schedule learning rate
        self.network.optimizer_G.param_groups[0]['lr'] = lr_linear_schedule(
            epoch,
            epoch_start=self.options.scheduler_epoch_range[0],
            epoch_end=self.options.scheduler_epoch_range[1],
            lr_base=self.options.lr_g,
            lr_end=self.options.scheduler_lr_g_end
        )
        self.network.optimizer_D.param_groups[0]['lr'] = lr_linear_schedule(
            epoch,
            epoch_start=self.options.scheduler_epoch_range[0],
            epoch_end=self.options.scheduler_epoch_range[1],
            lr_base=self.options.lr_d,
            lr_end=self.options.scheduler_lr_d_end
        )

        # LOG LEARNING RATES
        self.logger.log_info(f'Update learning rates: LR_G = {self.network.optimizer_G.param_groups[0]["lr"]:.8f} LR D = {self.network.optimizer_D.param_groups[0]["lr"]:.8f}')
        self.logger.log_scalar('LR_G', self.network.optimizer_G.param_groups[0]["lr"], epoch)
        self.logger.log_scalar('LR_D', self.network.optimizer_D.param_groups[0]["lr"], epoch)

        # SAVE MODEL (EPOCH)
        self.network.save_model(self.network.G, self.network.optimizer_G, epoch, self.iterations, self.options, time_for_name=self.run_start)
        self.network.save_model(self.network.D, self.network.optimizer_D, epoch, self.iterations, self.options, time_for_name=self.run_start)


    def _train_epoch(self, epoch):
        set_adaptive_strategy(False)
        set_gen_counter(0)
        loss_G, loss_D = None, None
        loss_G_prev, loss_D_prev = None, None

        for batch_num, batch in enumerate(self.data_loader_train):
            batch_start = datetime.now()

            # Fixed update strategy
            if not is_adaptive_strategy():
                d_iters = self.options.d_iters if self.options.gan_type == 'wgan-gp' else 2
                set_gen_active((batch_num + 1) % d_iters == 0)

                if is_gen_active():
                    images_generated, loss_G, losses_G_dict = self.network.forward_G(batch)
                    gen_counter_inc()
                else:
                    loss_D, losses_D_dict = self.network.forward_D(batch)

                # Start adaptive strategy after 3 fixed update cycles
                if self.options.loss_coeff > 0 and (batch_num + 1) % (d_iters * 3) == 0:
                    set_adaptive_strategy(True)

            # Adaptive update strategy
            if self.options.loss_coeff > 0:
                # Initialize loss change ratios r_d, r_g
                if batch_num == 0:
                    r_d, r_g = 1, 1

                # Initialize prev loss to current for first iteration
                if is_gen_active() and loss_G_prev is None:
                    loss_G_prev = loss_G
                elif not is_gen_active() and loss_D_prev is None:
                    loss_D_prev = loss_D

                if r_d > self.options.loss_coeff * r_g:
                    if is_adaptive_strategy():
                        set_gen_active(False)
                        loss_D, losses_D_dict = self.network.forward_D(batch)
                        loss_G = self.network.get_loss_G(batch)
                else:
                    if is_adaptive_strategy():
                        set_gen_active(True)
                        gen_counter_inc()
                        images_generated, loss_G, losses_G_dict = self.network.forward_G(batch)
                        loss_D = self.network.get_loss_D(batch['image2'], images_generated)

                if loss_G_prev is not None and loss_D_prev is not None:
                    r_g, r_d = abs((loss_G - loss_G_prev) / loss_G_prev), abs((loss_D - loss_D_prev) / loss_D_prev)
                    loss_G_prev, loss_D_prev = loss_G, loss_D

            batch_end = datetime.now()

            # LOG UPDATE INTERVALS
            if is_gen_active():
                self.logger.log_scalar('Generator-Discriminator Updates', 1, self.iterations)
            else:
                self.logger.log_scalar('Generator-Discriminator Updates', 0, self.iterations)

            # LOG PROGRESS
            if loss_G is not None and loss_D is not None and (batch_num + 1) % d_iters == 0:
                self.logger.log_info(f'Epoch {epoch + 1}: [{str(batch_num + 1).zfill(len(str(len(self.data_loader_train))))}/{len(self.data_loader_train)}] | '
                                    f'Time: {batch_end - batch_start} | '
                                    f'Loss_G = {loss_G:.4f} Loss_D = {loss_D:.4f}')

                # LOG LOSSES G AND D
                self.logger.log_scalars(losses_G_dict, self.iterations)
                self.logger.log_scalars(losses_D_dict, self.iterations)

                if is_gen_active():
                    # LOG LATEST GENERATED IMAGE
                    images_real = batch['image2'].detach().clone()
                    images_fake = images_generated.detach().clone()
                    images_source = batch['image1'].detach().clone()
                    landmarks_target = batch['landmark2'].detach().clone()
                    images = torch.cat((images_source, landmarks_target, images_real, images_fake), dim=0)
                    self.logger.save_image(self.options.gen_dir, f'0_last_result', images, nrow=self.options.batch_size)

                    # LOG GENERATED IMAGES
                    if (not is_adaptive_strategy() and (batch_num + 1) % self.options.log_freq == 0) or (is_adaptive_strategy() and (get_gen_counter() + 1) > self.options.log_freq):
                        set_gen_counter(0)
                        self.logger.save_image(self.options.gen_dir, f't_{datetime.now():%Y%m%d_%H%M%S}', images, epoch=epoch, iteration=self.iterations, nrow=self.options.batch_size)
                        self.logger.log_image('Train/Generated', images, self.iterations, nrow=self.options.batch_size)
                        
                        # LOG EVALUATION METRICS
                        if self.options.test:
                            val_time_start = datetime.now()
                            ssim_train, fid_train = self.evaluate_metrics(images_real, images_fake, device=self.fid.device)
                            val_time_end = datetime.now()
                            self.logger.log_info(f'Validation: Time: {val_time_end - val_time_start} | SSIM = {ssim_train:.4f} | FID = {fid_train:.4f}')
                            self.logger.log_scalar('SSIM Train', ssim_train, self.iterations)
                            self.logger.log_scalar('FID Train', fid_train, self.iterations)
                            del images_generated, images_real, images_fake, images, images_source, landmarks_target, ssim_train, fid_train
                        else:
                            del images_generated, images_real, images_fake, images, images_source, landmarks_target
                    else:
                        del images_generated, images_real, images_fake, images, images_source, landmarks_target

            # # SAVE MODEL
            # if (batch_num + 1) % self.options.checkpoint_freq == 0:
            #     self.network.save_model(self.network.G, self.network.optimizer_G, self.network.scheduler_G, epoch, self.iterations, self.options, self.run_start)
            #     self.network.save_model(self.network.D, self.network.optimizer_D, self.network.scheduler_D, epoch, self.iterations, self.options, self.run_start)

            self.iterations += 1

        del loss_G, loss_D, loss_G_prev, loss_D_prev, r_g, r_d, losses_G_dict, losses_D_dict


    def evaluate_metrics(self, images_real, images_fake, device):
        ssim_score = calculate_ssim(images_fake.to(device), images_real.to(device))

        self.fid.calculate_activations(images_real.to(device), images_fake.to(device), batch_num=1)
        fid_score = self.fid.calculate_fid()

        return ssim_score.mean().item(), fid_score


# Global Tensors with shared memory. Needed for subprocesses of DataLoader
adaptive_strategy = torch.tensor([False])
gen_active = torch.tensor([False])
gen_counter = torch.tensor([0])
adaptive_strategy.share_memory_()
gen_active.share_memory_()
gen_counter.share_memory_()

def is_adaptive_strategy():
    global adaptive_strategy
    return adaptive_strategy

def set_adaptive_strategy(b: bool):
    global adaptive_strategy
    adaptive_strategy = torch.tensor([b])
    return adaptive_strategy

def is_gen_active():
    global gen_active
    return gen_active

def set_gen_active(b: bool):
    global gen_active
    gen_active = torch.tensor([b])
    return gen_active

def get_gen_counter():
    global gen_counter
    return gen_counter

def set_gen_counter(v):
    global gen_counter
    gen_counter = torch.tensor([v])
    return gen_counter

def gen_counter_inc():
    global gen_counter
    gen_counter += 1
    return gen_counter
