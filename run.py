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

def generate_images(experiments_root, options, logger):
    experiments_list = sorted(os.listdir(experiments_root))
    for experiment in experiments_list:
        checkpoint_dir = os.path.join(experiments_root, experiment, 'checkpoints')
        gen_dir = os.path.join(experiments_root, experiment, 'generated_test')
        models = sorted([f for f in os.listdir(checkpoint_dir) if 'Generator' in f])
        for model in models:
            model_path = os.path.join(checkpoint_dir, model)
            network = Network(logger, options, model_path=model_path)
            Test(logger, options, network).generate(gen_dir, epoch=network.continue_epoch)

def main():
    # mode: 'dataset', 'train', 'test', 'infer' or 'logs'
    mode = sys.argv[1]
    del sys.argv[1]
    description = 'Facial Reenactment'

    try:
        if mode == 'dataset':
            options = DatasetOptions(description=f'{description} Dataset')
            logger = Logger(options)
            logger.init_writer()
            if options.num_workers > 0: torch.multiprocessing.set_start_method('spawn')
            Preprocess(logger, options).preprocess_dataset()

        elif mode == 'train':
            options = TrainOptions(description=f'{description} Training')
            logger = Logger(options)
            logger.init_writer()
            if options.num_workers > 0: torch.multiprocessing.set_start_method('spawn')
            Train(logger, options)
            if options.plots is not None:
                LogsExtractor(logger, options, options.log_dir, after_train=True)

        elif mode == 'test':
            options = TestOptions(description=f'{description} Testing')
            logger = Logger(options)
            logger.init_writer()
            if options.num_workers_test > 0: torch.multiprocessing.set_start_method('spawn')

            # Test single model
            if options.model is not None:
                Test(logger, options, Network(logger, options, model_path=options.model)).test()

            # Test multiple models
            else:
                models = sorted([f for f in os.listdir(options.checkpoint_dir) if 'Generator' in f])
                for model in models:
                    network = Network(logger, options, model_path=os.path.join(options.checkpoint_dir, model))
                    Test(logger, options, network).test(network.continue_epoch)
                # generate_images(EXPERIMENTS_ROOT_DIR, options, logger)

        elif mode == 'infer':
            options = TestOptions(description=f'{description} Inference')
            logger = Logger(options)
            infer = Infer(logger, options, options.model_path)

            if options.target.endswith('.mp4'):
                infer.from_video()
            else:
                infer.from_image()

        elif mode == 'logs':
            options = LogsOptions(description=f'{description} Logs-Extracting')
            logger = Logger(options)
            logger.init_writer()
            LogsExtractor(logger, options, options.logs_dir)

        else:
            print('invalid command')
    except Exception as e:
        print(f'Something went wrong: {e}')
        raise e

if __name__ == '__main__':
    main()
