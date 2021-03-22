from configs import Options

class TestOptions(Options):
    def __init__(self, description):
        super(TestOptions, self).__init__(description)
        self._init_parser()
        self._parse_args()


    def _init_parser(self):
        # ARGUMENTS: OPTIONS
        self.parser.add_argument('--device', type=str, default=self.config['device'], help='Whether to run the model on GPU or CPU.')
        self.check_error(self.config, 'device', ['cuda', 'cpu'])

        self.parser.add_argument('--pin_memory', action='store_false' if self.config['pin_memory'] else 'store_true')

        self.parser.add_argument('--num_workers', type=int, default=self.config['num_workers'])

        # ARGUMENTS: DIRECTORIES
        self.parser.add_argument('--log_dir', type=str, default=self.config['paths']['log_dir'], help='Path where logs will be saved.')

        self.parser.add_argument('--output_dir', type=str, default=self.config['paths']['output_dir'], help='Path where output will be saved.')

        # ARGUMENTS: INPUTS
        self.parser.add_argument('--source', type=str, default=None, help='Path for source image (identity to be preserved).')

        self.parser.add_argument('--target', type=str, default=None, help='Path for target image.')

        self.parser.add_argument('--model', type=str, default=None, help='Path to the generator model.')

        # ARGUMENTS: DATASET
        self.parser.add_argument('--batch_size_test', type=int, default=self.config['test']['batch_size_test'])

        self.parser.add_argument('--shuffle_test', action='store_false' if self.config['test']['shuffle_test'] else 'store_true')

        self.parser.add_argument('--log_freq_test', type=int, default=self.config['test']['log_freq_test'])

        self.parser.add_argument('--normalize', nargs='+', default=self.config['dataset']['normalize'], type=float, help='Image normalization: mean, std')        
