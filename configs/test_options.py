from configs import config, Options

class TestOptions(Options):
    def __init__(self, description):
        super(TestOptions, self).__init__(description)
        self._init_parser()
        self.args = self._parse_args()


    def _init_parser(self):
        self.parser.add_argument('--pin_memory', action='store_true')
        self.parser.add_argument('--num_workers', type=int, default=0)
        self.parser.add_argument('--source', type=str, required=True,
                                            help='Path for source image (identity to be preserved).')
        self.parser.add_argument('--target', type=str, required=True,
                                            help='Path for target image.')
        self.parser.add_argument('--model', type=str, required=True,
                                            help='Path to the generator model.')
        self.parser.add_argument('--device', nargs='?', default='cuda', const='cuda', choices=['cuda', 'cpu'],
                                            help='Whether to run the model on GPU or CPU.')
        self.parser.add_argument('--log_dir', type=str, default=config.LOG_DIR,
                                            help='Path where logs will be saved.')
        self.parser.add_argument('--output', type=str, default=config.TEST_DIR,
                                            help='Path where output will be saved.')
