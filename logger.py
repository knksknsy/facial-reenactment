import os
import sys
import logging
import cv2

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from configs.options import Options

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


    def init_writer(self, filename: str = None):
        if filename is not None:
            self.writer = SummaryWriter(os.path.join(self.options.log_dir, filename))
        else:
            self.writer = SummaryWriter(self.options.log_dir)


    def log_info(self, message: str):
        logging.info(message)


    def log_debug(self, message: str):
        logging.debug(message)


    def log_error(self, message: str):
        logging.error(message)


    def log_scalar(self, tag: str, y_value, x_value):
        self.writer.add_scalar(tag, y_value, x_value)
        self.writer.flush()


    def save_image(self, path, filename, image, epoch=None, iteration=None):
        if not os.path.isdir(path):
            os.makedirs(path)

        grid = self._get_grid(image, bgr=True)

        if epoch is not None and iteration is not None:
            filename = f'e_{epoch}_b{iteration}_{filename}'
        
        cv2.imwrite(os.path.join(path, filename), grid)


    def show_image(self, image):
        grid = self._get_grid(image, bgr=True)

        cv2.imshow('Image Preview', grid)


    def log_image(self, tag: str, image, label):
        grid = self._get_grid(image)
        self.writer.add_images(tag, grid, label)
        self.writer.flush()


    def _get_grid(self, data, bgr=False, nrow=4, normalize=True, range=(-1,1)):
        grid = make_grid(data.clone().detach(), nrow, normalize, range)
        if bgr:
            grid.cpu().numpy().transpose(1, 2, 0)
        return grid
