import os
import sys
import logging
import cv2

import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from configs.options import Options
from utils.transforms import denormalize

class Logger():
    def __init__(self, options: Options):
        self.options = options
        self._init_logger()


    def _init_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(self.options.log_dir, f'{datetime.now():%Y-%m-%d}.log'),
            format='[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


    def init_writer(self, filename: str = None, filename_suffix: str = ''):
        if filename is not None:
            self.writer = SummaryWriter(os.path.join(self.options.log_dir, filename), filename_suffix=filename_suffix)
        else:
            self.writer = SummaryWriter(self.options.log_dir, filename_suffix=filename_suffix)


    def log_info(self, message: str):
        logging.info(message)


    def log_infos(self, infos: dict):
        messages = []
        for key, value in infos.items():
            messages.append(f'{key} = {value:.6f}')
        message = ' | '.join(messages)
        logging.info(message)


    def log_debug(self, message: str):
        logging.debug(message)


    def log_error(self, message: str):
        logging.error(message)


    def log_scalar(self, tag: str, y_value, x_value, tag_prefix=None):
        if tag_prefix is not None:
            tag = f'{tag_prefix}_{tag}'
        self.writer.add_scalar(tag, y_value, x_value)
        self.writer.flush()


    def log_scalars(self, scalars: dict, x_value, tag_prefix=None):
        scalars = {k: v for k, v in scalars.items() if v is not None}
        for key, value in scalars.items():
            if tag_prefix is not None:
                key = f'{tag_prefix}_{key}'
            self.writer.add_scalar(tag=key, scalar_value=value, global_step=x_value)
        self.writer.flush()


    def save_image(self, path: str, filename: str, image, nrow=4, ext: str='.png', epoch: int=None, iteration: int=None, ret_image=False):
        grid = self._get_grid(image, nrow=nrow, as_tensor=False)

        if ret_image:
            return grid

        if epoch is not None and iteration is not None:
            filename = f'{filename}_e_{str(epoch).zfill(3)}_b{str(iteration).zfill(8)}{ext}'
        else:
            filename = f'{filename}{ext}'

        cv2.imwrite(os.path.join(path, filename), grid)


    def show_image(self, image, nrow=4):
        grid = self._get_grid(image, nrow=nrow, as_tensor=False)

        cv2.imshow('Image Preview', grid)


    def log_image(self, tag: str, image, label, nrow=4):
        grid = self._get_grid(image, nrow=nrow)
        self.writer.add_image(tag, grid, label)
        self.writer.flush()


    def _get_grid(self, data, nrow=4, as_tensor=True):
        data = denormalize(data, self.options.normalize[0], self.options.normalize[1])
        grid = make_grid(data, nrow)

        if as_tensor:
            grid = grid[[2, 1, 0],:]
        else:
            grid = grid.cpu().numpy().transpose(1, 2, 0) * 255.0
            grid = grid.clip(0.0, 255.0).astype(np.uint8)

        return grid
