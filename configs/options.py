import argparse
import torch
import os
import yaml

from abc import ABC, abstractmethod

class Options(ABC):
    def __init__(self, description):
        self.description = description
        self.parser = argparse.ArgumentParser(description=self.description)
        self.config = self._load_config()


    @abstractmethod
    def _init_parser(self):
        pass


    def _parse_args(self):
        self.args = self.parser.parse_args()
        if hasattr(self.args, 'device'):
            self.device = 'cuda' if (torch.cuda.is_available() and self.args.device == 'cuda') else 'cpu'
        # Set default values from config file
        self._set_properties(self.config)
        # Override default values
        self._set_properties(vars(self.args))

    
    def check_error(self, config, arg, choices):
        if config[arg] not in choices:
            self.parser.error(f'Ilegal value for --{arg}={config[arg]}! Allowed choices: {choices}')


    def _load_config(self):
        self.parser.add_argument('--config', type=str, required=True,  help='Path to YAML config file.')
        args, unkown = self.parser.parse_known_args()

        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config


    def _set_properties(self, d):
        for k, v in d.items():
            if k == 'config':
                continue
            if isinstance(v, dict):
                self._set_properties(v)
            else:
                setattr(self, k, v)
                if '_dir' in k and not os.path.isdir(v):
                    os.makedirs(v)


    def __str__(self):
        output = ''

        for k, v in vars(self.args).items():
            output += f'{k:<20s}:\t{v}\n'

        return output
