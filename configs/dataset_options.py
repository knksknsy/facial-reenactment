from configs import Options

class DatasetOptions(Options):
    def __init__(self, description):
        super(DatasetOptions, self).__init__(description)
        self._init_parser()
        self._parse_args()


    def _init_parser(self):
        # ARGUMENTS: OPTIONS
        self.parser.add_argument('--device', nargs='?', default=self.config['device'], const=self.config['device'],
                                            choices=self.config['device_options'],
                                            help='Whether to run the model on GPU or CPU.')
        # ARGUMENTS: DIRECTORIES
        self.parser.add_argument('--log_dir', type=str, default=self.config['paths']['log_dir'],
                                            help='Path where logs will be saved.')

        # ARGUMENTS: DATASET
        self.parser.add_argument('--source', type=str, required=True,
                                            help='Path to the source folder where the raw VoxCeleb2 dataset is located.')
        self.parser.add_argument('--output', type=str, required=True,
                                            help='Path to the folder where the pre-processed dataset will be stored.')

        # ARGUMENTS: INPUTS
        self.parser.add_argument('--csv', type=str, required=True,
                                            help='Path to where the CSV file will be saved.')
        self.parser.add_argument('--num_frames', type=int, default=self.config['preprocessing']['num_frames']+1,
                                            help='Number of frames + 1 to extract from a video.')
        self.parser.add_argument('--num_videos', type=int, default=self.config['preprocessing']['num_videos'],
                                            help='Number of videos from the dataset to process. Providing 0 will pre-process all videos.')
        self.parser.add_argument('--overwrite_videos', action='store_false' if self.config['preprocessing']['overwrite_videos'] else 'store_true',
                                            help='Add this flag to overwrite already pre-processed files. The default functionality'
                                            'is to ignore videos that have already been pre-processed.')
        self.parser.add_argument('--vox_ids', nargs='+', type=str, help='Voxceleb2 ids to be processes')
