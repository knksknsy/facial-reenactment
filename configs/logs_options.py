from configs import Options

class LogsOptions(Options):
    def __init__(self, description):
        super(LogsOptions, self).__init__(description)
        self._init_parser()
        self._parse_args()


    def _init_parser(self):
        # ARGUMENTS: OPTIONS
        self.parser.add_argument('--log_dir', type=str, required=True, help='Path to the tensorboard events directory.')
        self.parser.add_argument('--output_dir', type=str, required=True, help='Path where the outputs (Metrics-CSV, Images) will be saved.')
