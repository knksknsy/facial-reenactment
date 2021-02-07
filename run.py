import sys

from configs import TrainOptions, TestOptions, DatasetOptions
from training import Train
from testing import Test
from dataset import Preprocess
from models import Network
from logger import Logger

def main():
    # mode: 'dataset', 'train', or 'test'
    mode = sys.argv[1]
    description = 'Facial Reenactment'

    try:
        if mode == 'dataset':
            options = DatasetOptions(description=f'{description} Dataset')
            logger = Logger(options)
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

        else:
            print('invalid command')
    except Exception as e:
        print(f'Something went wrong: {e}')
        raise e


if __name__ == '__main__':
    main()
