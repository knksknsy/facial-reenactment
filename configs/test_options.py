import json
from utils.utils import Method
from configs import Options

class TestOptions(Options):
    def __init__(self, description: str, method):
        super(TestOptions, self).__init__(description, method)
        self._init_parser()
        self._parse_args()

        if self.plots is not None:
            self.plots = self._load_plots_config()


    def _load_plots_config(self):
        with open(self.plots) as f:
            plots = json.load(f)
        return plots


    def _init_parser(self):
        # ARGUMENTS: OPTIONS
        self.parser.add_argument('--device', type=str, default=self.config['device'], help='Whether to run the model on GPU or CPU.')
        self.check_error(self.config, 'device', ['cuda', 'cpu'])

        self.parser.add_argument('--pin_memory', action='store_false' if self.config['pin_memory'] else 'store_true')

        self.parser.add_argument('--num_workers', type=int, default=self.config['num_workers'])

        self.parser.add_argument('--tag_prefix', type=str, default=self.config['test']['tag_prefix'], help='Prefix for Tensorboard loggings.')

        self.parser.add_argument('--seed', type=int, default=self.config['train']['seed'])

        # ARGUMENTS: DIRECTORIES
        self.parser.add_argument('--log_dir', type=str, default=self.config['paths']['log_dir'], help='Path where logs will be saved.')

        self.parser.add_argument('--output_dir', type=str, default=self.config['paths']['output_dir'], help='Path where output will be saved.')

        self.parser.add_argument('--dataset_test', type=str, default=self.config['dataset']['dataset_test'], help='Path to the pre-processed dataset for testing.')

        self.parser.add_argument('--csv_test', type=str, default=self.config['dataset']['csv_test'], help='Path to CSV file needed for torch.utils.data.Dataset to load data for testing.')

        # ARGUMENTS: PLOTS DIRECTORY
        self.parser.add_argument('--plots', type=str, help='Path to the plots.json configuration file.')

        # ARGUMENTS: INPUTS
        self.parser.add_argument('--source', type=str, default=None, help='Path for source image (identity to be preserved).')

        self.parser.add_argument('--target', type=str, default=None, help='Path for target image.')

        self.parser.add_argument('--model', type=str, default=None, help='Path to the generator model.')

        # ARGUMENTS: DATASET
        self.parser.add_argument('--batch_size_test', type=int, default=self.config['test']['batch_size_test'])

        self.parser.add_argument('--shuffle_test', action='store_true' if self.config['test']['shuffle_test'] else 'store_false')

        self.parser.add_argument('--log_freq_test', type=int, default=self.config['test']['log_freq_test'])

        self.parser.add_argument('--num_workers_test', type=int, default=self.config['test']['num_workers_test'])

        self.parser.add_argument('--normalize', nargs='+', default=self.config['dataset']['normalize'], type=float, help='Image normalization: mean, std')

        self.parser.add_argument('--image_size', default=self.config['dataset']['image_size'], type=int, help='Image size')

        self.parser.add_argument('--channels', default=self.config['dataset']['channels'], type=int, help='Image channels')

        self.parser.add_argument('--padding', type=int, default=self.config['preprocessing']['padding'], help='Padding size')

        if self.method == Method.CREATION:
            self.parser.add_argument('--landmark_type', type=str, default=self.config['train']['landmark_type'], help='Facial landmark type: boundary | keypoint')
            self.check_error(self.config['train'], 'landmark_type', ['boundary', 'keypoint'])

            self.parser.add_argument('--conv_blocks_d', default=self.config['train']['conv_blocks_d'], type=int, help='Number of convolutional layers in discriminator: 4 | 6')

        if self.method == Method.DETECTION:
            self.parser.add_argument('--threshold', default=self.config['train']['threshold'], type=float, help='Threshold for binary classification.')
            
            self.parser.add_argument('--batch_size_class', default=self.config['train']['batch_size_class'], type=int, help='Batch size')

            self.parser.add_argument('--len_feature', default=self.config['train']['len_feature'], type=int, help='Length of feature vector.')

            self.parser.add_argument('--hidden_layer_num_features', default=self.config['train']['hidden_layer_num_features'], type=int, help='Length of hidden layer of classifier.')

            self.parser.add_argument('--l_mask', default=self.config['train']['loss_weights']['l_mask'], type=float, help='Mask loss.')

            self.parser.add_argument('--l_mask_sv', default=self.config['train']['loss_weights']['l_mask_sv'], type=float, help='Mask loss regression type: 1=supervised | 2=unsupervised')

            self.parser.add_argument('--loss_type', type=str, default=self.config['train']['loss_type'], help='Loss type for feature extraction: contrastive | triplet')
            self.check_error(self.config['train'], 'loss_type', ['contrastive', 'triplet'])

            # ARGUMENTS: DATASET
            self.parser.add_argument('--mask_size', default=self.config['dataset']['mask_size'], type=int, help='Mask size')
