import os
import sys
import logging
from datetime import datetime

from configs.options import Options
from configs import TrainOptions, TestOptions, DatasetOptions
from training import Train
from testing import Test
from dataset import Preprocess

def main():
    # mode: 'dataset', 'train', or 'test'
    mode = sys.argv[1]
    description = 'Facial Reenactment'

    try:
        if mode == 'dataset':
            options = DatasetOptions(description=f'{description} Dataset')
            setup_logger(options)
            Preprocess(options)

        elif mode == 'train':
            options = TrainOptions(description=f'{description} Training')
            setup_logger(options)
            Train(options)

        elif mode == 'test':
            options = TestOptions(description=f'{description} Testing')
            setup_logger(options)
            Test(options, training=False)

        else:
            print('invalid command')
    except Exception as e:
        logging.error(f'Something went wrong: {e}')
        raise e


def setup_logger(options: Options):
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(options.log_dir, f'{datetime.now():%Y-%m-%d}.log'),
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


if __name__ == '__main__':
    main()
