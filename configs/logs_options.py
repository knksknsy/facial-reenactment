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

        self.parser.add_argument('--overwrite_csv', action='store_false' if self.config['logs']['overwrite_csv'] else 'store_true')

        self.parser.add_argument('--overwrite_plot', action='store_false' if self.config['logs']['overwrite_plot'] else 'store_true')

        self.parser.add_argument('--single_experiment', action='store_false' if self.config['logs']['single_experiment'] else 'store_true')
