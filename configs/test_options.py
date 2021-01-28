from configs import config, Options

class TestOptions(Options):
    def __init__(self, description):
        super(TestOptions, self).__init__(description)
        self._init_parser()
        self.args = self._parse_args()


    def _init_parser(self):
        self.parser.add_argument("--image", type=str, required=True,
                                            help="Path for image to be reenacted.")
        self.parser.add_argument("--landmarks", type=str, required=True,
                                            help="Path for landmarks to reenact image.")
        self.parser.add_argument("--model", type=str, required=True,
                                            help="Path to the generator model.")
        self.parser.add_argument("--device", nargs='?', default='cuda', const='cuda', choices=['cuda', 'cpu'],
                                            help="Whether to run the model on GPU or CPU.")
        self.parser.add_argument("--log_dir", type=str, default=config.LOG_DIR,
                                            help="Path where logs will be saved.")
