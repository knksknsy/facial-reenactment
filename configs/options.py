import argparse
import torch
import os

from abc import ABC, abstractmethod

class Options(ABC):
    def __init__(self, description):
        self.description = description
        self.parser = argparse.ArgumentParser(description=self.description)


    @abstractmethod
    def _init_parser(self):
        pass


    def _parse_args(self):
        args, unknown = self.parser.parse_known_args()
        self.device = 'cuda' if (torch.cuda.is_available() and args.device == 'cuda') else 'cpu'

        for k, v in vars(args).items():
            setattr(self, k, v)
            if '_dir' in k and not os.path.exists(v):
                os.makedirs(v)

        return args

    def __str__(self):
        output = ''

        for k, v in vars(self.args).items():
            output += f'{k:<20s}:\t{v}\n'

        return output
