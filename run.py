import os
import sys
import logging
from datetime import datetime

from configs import config, Options
from training import Train
from dataset import Preprocess

def main():
    # PARSE ARGUMENTS
    options = Options()

    # SETUP LOGGER
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(options.args.log_dir, f'{datetime.now():%Y-%m-%d}.log'),
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    try:
        if options.args.subcommand == "train":
            Train(options)
        elif options.args.subcommand == "dataset":
            Preprocess(options)
        else:
            print("invalid command")
    except Exception as e:
        logging.error(f'Something went wrong: {e}')
        raise e


if __name__ == '__main__':
    main()
