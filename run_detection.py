import os
import sys
from utils.utils import Method
import torch

from utils import Mode
from configs import TrainOptions, TestOptions, DatasetOptions, LogsOptions

from loggings.logger import Logger
from loggings.extract import LogsExtractor
from detection.training import Trainer
from detection.testing import Tester
from detection.inference import Infer
from detection.dataset import PreprocessFaceForensics
from detection.models import Network


def main(mode, method, description: str):
    try:
        ##### DATASET PREPROCESSING #####
        if mode == Mode.DATASET:
            options = DatasetOptions(description=f'{description} Dataset', method=method)
            logger = Logger(options)
            logger.init_writer()
            if options.num_workers > 0: torch.multiprocessing.set_start_method('spawn')

            PreprocessFaceForensics(logger, options, options.methods).preprocess()


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
                Tester(logger, options, network).test_classification(network.continue_epoch)

            # Test multiple models
            else:
                models = sorted([f for f in os.listdir(options.checkpoint_dir) if 'SiameseResNet' in f])
                for model in models:
                    network = Network(logger, options, model_path=os.path.join(options.checkpoint_dir, model))
                    Tester(logger, options, network).test_classification(network.continue_epoch)


        ##### INFERENCE #####
        elif mode == Mode.INFER:
            options = TestOptions(description=f'{description} Inference', method=method)
            logger = Logger(options)
            infer = Infer(logger, options, options.source,  options.model)

            if options.source.endswith('.mp4'):
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
# e.g.: $python run_detection.py train [arguments]
if __name__ == '__main__':
    mode = sys.argv[1].upper()
    del sys.argv[1]
    main(Mode[mode], Method.DETECTION, description='Facial Reenactment')
