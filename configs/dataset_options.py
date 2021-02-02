from configs import config, Options

class DatasetOptions(Options):
    def __init__(self, description):
        super(DatasetOptions, self).__init__(description)
        self._init_parser()
        self.args = self._parse_args()


    def _init_parser(self):
        self.parser.add_argument('--source', type=str, required=True,
                                            help='Path to the source folder where the raw VoxCeleb2 dataset is located.')
        self.parser.add_argument('--output', type=str, required=True,
                                            help='Path to the folder where the pre-processed dataset will be stored.')
        self.parser.add_argument('--csv', type=str, required=True,
                                            help='Path to where the CSV file will be saved.')
        self.parser.add_argument('--frames', type=int, default=config.K+1,
                                            help='Number of frames + 1 to extract from a video.')
        self.parser.add_argument('--size', type=int, default=0,
                                            help='Number of videos from the dataset to process. Providing 0 will pre-process all videos.')
        self.parser.add_argument('--device', nargs='?', default='cuda', const='cuda', choices=['cuda', 'cpu'],
                                            help='Whether to run the model (face_alignment) on GPU or CPU.')
        self.parser.add_argument('--overwrite', action='store_true',
                                            help='Add this flag to overwrite already pre-processed files. The default functionality'
                                            'is to ignore videos that have already been pre-processed.')
        self.parser.add_argument('--log_dir', type=str, default=config.LOG_DIR,
                                            help='Path where logs will be saved.')
