import json
from configs import Options

class LogsOptions(Options):
    def __init__(self, description):
        super(LogsOptions, self).__init__(description)
        self._init_parser()
        self._parse_args()
        self.plots = self._load_plots_config()


    def _load_plots_config(self):
        with open(self.plots) as f:
            plots = json.load(f)
        return plots


    def _init_parser(self):
        # ARGUMENTS: OPTIONS
        self.parser.add_argument('--logs_dir', type=str, required=True, help='Path to the tensorboard events directory.')

        self.parser.add_argument('--plots', type=str, required=True, help='Path to the plots.json configuration file.')

        self.parser.add_argument('--overwrite_logs', action='store_true')
