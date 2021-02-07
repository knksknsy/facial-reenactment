from configs import Options

class TestOptions(Options):
    def __init__(self, description):
        super(TestOptions, self).__init__(description)
        self._init_parser()
        self._parse_args()


    def _init_parser(self):
        # ARGUMENTS: OPTIONS
        self.parser.add_argument('--device', nargs='?', default=self.config['device'], const=self.config['device'],
                                            choices=self.config['device_options'],
                                            help='Whether to run the model on GPU or CPU.')
        self.parser.add_argument('--pin_memory', action='store_false' if self.config['pin_memory'] else 'store_true')
        self.parser.add_argument('--num_workers', type=int, default=self.config['num_workers'])

        # ARGUMENTS: DIRECTORIES
        self.parser.add_argument('--log_dir', type=str, default=self.config['paths']['log_dir'],
                                            help='Path where logs will be saved.')
        self.parser.add_argument('--output_dir', type=str, default=self.config['paths']['output_dir'],
                                            help='Path where output will be saved.')

        # ARGUMENTS: INPUTS
        self.parser.add_argument('--source', type=str, default=None,
                                            help='Path for source image (identity to be preserved).')
        self.parser.add_argument('--target', type=str, default=None,
                                            help='Path for target image.')
        self.parser.add_argument('--model', type=str, required=True,
                                            help='Path to the generator model.')

        # ARGUMENTS: DATASET
        self.parser.add_argument('--shuffle', action='store_false' if self.config['dataset']['shuffle'] else 'store_true')
        
