import os
import sys
import logging
from datetime import datetime

from configs import config, TrainOptions, DatasetOptions
from training import Train
from dataset import Preprocess

def main():
    # mode: 'dataset', 'train', or 'test'
    mode = sys.argv[1]

    # SETUP LOGGER
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(options.log_dir, f'{datetime.now():%Y-%m-%d}.log'),
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    try:
        if mode == "dataset":
            options = DatasetOptions()
            Preprocess(options)
        elif mode == "train":
            options = TrainOptions()
            Train(options)
        # elif mode == "test":
        #     options = TestOptions()
        #     Test(options)
        else:
            print("invalid command")
    except Exception as e:
        logging.error(f'Something went wrong: {e}')
        raise e


if __name__ == '__main__':
    main()
