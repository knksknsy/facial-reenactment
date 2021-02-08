import sys

from configs import TrainOptions, TestOptions, DatasetOptions, LogsOptions
from training import Train
from testing import Test
from dataset import Preprocess
from models import Network
from loggings.logger import Logger
from loggings.extract import LogsExtractor

def main():
    # mode: 'dataset', 'train', 'test', or 'logs'
    mode = sys.argv[1]
    description = 'Facial Reenactment'

    try:
        if mode == 'dataset':
            options = DatasetOptions(description=f'{description} Dataset')
            logger = Logger(options)
            logger.init_writer()
            Preprocess(logger, options)

        elif mode == 'train':
            options = TrainOptions(description=f'{description} Training')
            logger = Logger(options)
            logger.init_writer()
            Train(logger, options)

        elif mode == 'test':
            options = TestOptions(description=f'{description} Testing')
            logger = Logger(options)
            logger.init_writer()

            # Benchmark
            if options.source is None and options.target is None:
                Test(logger, options, Network(logger, options)).test()
            # Video mode
            elif '.mp4' in options.target:
                Test(logger, options).from_video()
            # Image mode
            else:
                Test(logger, options).from_image()

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
