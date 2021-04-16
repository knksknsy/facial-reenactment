import json
from utils.utils import Method
from configs import Options

class LogsOptions(Options):
    def __init__(self, description:str, method):
        super(LogsOptions, self).__init__(description, method)
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

        self.parser.add_argument('--output_dir', type=str, default=self.config['paths']['output_dir'], help='Path where output will be saved.')

        self.parser.add_argument('--plots', type=str, required=True, help='Path to the plots.json configuration file.')

        self.parser.add_argument('--overwrite_csv', action='store_false' if self.config['logs']['overwrite_csv'] else 'store_true')

        self.parser.add_argument('--overwrite_plot', action='store_false' if self.config['logs']['overwrite_plot'] else 'store_true')

        self.parser.add_argument('--overwrite_video', action='store_false' if self.config['logs']['overwrite_video'] else 'store_true')

        # ARGUMENTS: DATASET
        self.parser.add_argument('--normalize', nargs='+', default=self.config['dataset']['normalize'], type=float, help='Image normalization: mean, std')

        self.parser.add_argument('--image_size', default=self.config['dataset']['image_size'], type=int, help='Image size')

        self.parser.add_argument('--channels', default=self.config['dataset']['channels'], type=int, help='Image channels')

        self.parser.add_argument('--padding', type=int, default=self.config['preprocessing']['padding'], help='Padding size')

        if self.method == Method.CREATION:
            self.parser.add_argument('--landmark_type', type=str, default=self.config['train']['landmark_type'], help='Facial landmark type: boundary | keypoint')
            self.check_error(self.config['train'], 'landmark_type', ['boundary', 'keypoint'])