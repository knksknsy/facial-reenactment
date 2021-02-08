import os
import sys
import logging
import cv2

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from configs.options import Options
from dataset.utils import denormalize

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


    def save_image(self, path: str, filename: str, image, ext: str='.png', epoch: int=None, iteration: int=None):
        grid = self._get_grid(image, as_tensor=False)

        if epoch is not None and iteration is not None:
            filename = f'{filename}_e_{str(epoch).zfill(3)}_b{str(iteration).zfill(7)}{ext}'
        else:
            filename = f'{filename}{ext}'

        cv2.imwrite(os.path.join(path, filename), grid)


    def show_image(self, image):
        grid = self._get_grid(image, as_tensor=False)

        cv2.imshow('Image Preview', grid)


    def log_image(self, tag: str, image, label):
        grid = self._get_grid(image)
        self.writer.add_image(tag, grid, label)
        self.writer.flush()


    def _get_grid(self, data, nrow=4, denorm=(0.5, 0.5), as_tensor=True):
        if denorm is not None:
            data = denormalize(data, denorm[0], denorm[1])

        grid = make_grid(data, nrow)

        if not as_tensor:
            grid = grid.cpu().numpy().transpose(1, 2, 0) * 255.0
        else:
            grid = grid[[2, 1, 0],:]

        return grid
