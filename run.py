import os
import sys
import torch

from configs import TrainOptions, TestOptions, DatasetOptions, LogsOptions
from training import Train
from testing import Test
from inference import Infer
from dataset import Preprocess
from models import Network
from loggings.logger import Logger
from loggings.extract import LogsExtractor

def main():
    # mode: 'dataset', 'train', 'test', 'infer' or 'logs'
    mode = sys.argv[1]
    description = 'Facial Reenactment'

    try:
        if mode == 'dataset':
            options = DatasetOptions(description=f'{description} Dataset')
            logger = Logger(options)
            logger.init_writer()
            if options.num_workers > 0: torch.multiprocessing.set_start_method('spawn')
            Preprocess(logger, options)

        elif mode == 'train':
            options = TrainOptions(description=f'{description} Training')
            logger = Logger(options)
            logger.init_writer()
            if options.num_workers > 0: torch.multiprocessing.set_start_method('spawn')
            Train(logger, options)

        elif mode == 'test':
            options = TestOptions(description=f'{description} Testing')
            logger = Logger(options)
            logger.init_writer()
            if options.num_workers > 0: torch.multiprocessing.set_start_method('spawn')

            # Test single model
            if options.model is not None:
                Test(logger, options, Network(logger, options, model_path=options.model)).test()

            # Test multiple models
            else:
                models = sorted([f for f in os.listdir(options.checkpoint_dir) if 'Generator' in f])
                for epoch, model in enumerate(models):
                    network = Network(logger, options, model_path=model)
                    Test(logger, options, network).test(epoch)

        elif mode == 'infer':
            options = TestOptions(description=f'{description} Testing')
            logger = Logger(options)
            infer = Infer(logger, options, options.model_path)

            # Video mode
            if '.mp4' in options.target:
                infer.from_video()
            # Image mode
            else:
                infer.from_image()

        elif mode == 'logs':
            options = LogsOptions(description=f'{description} Logs-Extracting')
            logger = Logger(options)
            logger.init_writer()
            LogsExtractor(logger, options)

        else:
            print('invalid command')
    except Exception as e:
        print(f'Something went wrong: {e}')
        raise e

if __name__ == '__main__':
    main()
