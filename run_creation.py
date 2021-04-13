import os
import sys
import torch

from utils import Mode, Method
from configs import TrainOptions, TestOptions, DatasetOptions, LogsOptions

from loggings.logger import Logger
from loggings.extract import LogsExtractor
from creation.training import Trainer
from creation.testing import Tester
from creation.models import Network
from creation.inference import Infer
from creation.dataset import PreprocessVoxCeleb

# def generate_images(experiments_root, options, logger):
#     experiments_list = sorted(os.listdir(experiments_root))
#     for experiment in experiments_list:
#         checkpoint_dir = os.path.join(experiments_root, experiment, 'checkpoints')
#         gen_dir = os.path.join(experiments_root, experiment, 'generated_test')
#         models = sorted([f for f in os.listdir(checkpoint_dir) if 'Generator' in f])
#         for model in models:
#             model_path = os.path.join(checkpoint_dir, model)
#             network = Network(logger, options, model_path=model_path)
#             Tester(logger, options, network).generate(gen_dir, epoch=network.continue_epoch)


def main(mode, method, description: str):
    try:
        ##### DATASET PREPROCESSING #####
        if mode == Mode.DATASET:
            options = DatasetOptions(description=f'{description} Dataset', method=method)
            logger = Logger(options)
            logger.init_writer()
            if options.num_workers > 0: torch.multiprocessing.set_start_method('spawn')

            if hasattr(options, 'vox_ids'):
                PreprocessVoxCeleb(logger, options).preprocess_by_ids(options.vox_ids)
            else:
                PreprocessVoxCeleb(logger, options).preprocess()


        ##### TRAINING #####
        elif mode == Mode.TRAIN:
            options = TrainOptions(description=f'{description} Training', method=method)
            logger = Logger(options)
            logger.init_writer()
            if options.num_workers > 0: torch.multiprocessing.set_start_method('spawn')

            Trainer(logger, options).start()
            if options.plots is not None:
                LogsExtractor(logger, options, options.log_dir, multiples=False, video_per_model=True, method=method).start()


        ##### TESTING #####
        elif mode == Mode.TEST:
            options = TestOptions(description=f'{description} Testing', method=method)
            logger = Logger(options)
            logger.init_writer()
            if options.num_workers_test > 0: torch.multiprocessing.set_start_method('spawn')

            # Test single model
            if options.model is not None:
                network = Network(logger, options, model_path=options.model)
                Tester(logger, options, network).test(network.continue_epoch)

            # Test multiple models
            else:
                # generate_images(EXPERIMENTS_ROOT_DIR, options, logger)
                models = sorted([f for f in os.listdir(options.checkpoint_dir) if 'Generator' in f])
                for model in models:
                    network = Network(logger, options, model_path=os.path.join(options.checkpoint_dir, model))
                    Tester(logger, options, network).test(network.continue_epoch)


        ##### INFERENCE #####
        elif mode == Mode.INFER:
            options = TestOptions(description=f'{description} Inference', method=method)
            logger = Logger(options)
            infer = Infer(logger, options, options.source, options.target, options.model)

            if options.target.endswith('.mp4'):
                infer.from_video()
            else:
                infer.from_image()


        ##### LOGGING #####
        elif mode == Mode.LOGS:
            options = LogsOptions(description=f'{description} Logs-Extracting', method=method)
            logger = Logger(options)
            logger.init_writer()

            LogsExtractor(logger, options, options.logs_dir, multiples=True, video_per_model=False, method=method).start()

        else:
            print('invalid command')
    except Exception as e:
        print(f'Something went wrong: {e}')
        raise e

# mode: 'dataset', 'train', 'test', 'infer' or 'logs'
# e.g.: $python run_creation.py train [arguments]
if __name__ == '__main__':
    mode = sys.argv[1].upper()
    del sys.argv[1]
    main(Mode[mode], method=Method.CREATION, description='Facial Reenactment')
