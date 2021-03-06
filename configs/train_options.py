import json
from utils.utils import Method
from configs import Options

class TrainOptions(Options):
    def __init__(self, description: str, method):
        super(TrainOptions, self).__init__(description, method)
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

        self.parser.add_argument('--continue_id', type=str, default=self.config['train']['continue_id'], help='Id of the models to continue training.')

        self.parser.add_argument('--log_freq', type=int, default=self.config['train']['log_freq'], help='Frequency in which logs will be saved')

        self.parser.add_argument('--test', action='store_false' if self.config['train']['test'] else 'store_true', help='Model will be tested after each epoch.')

        self.parser.add_argument('--metrics', action='store_false' if self.config['train']['metrics'] else 'store_true', help='Evaluations will be calculated for train set during training.')

        self.parser.add_argument('--seed', type=int, default=self.config['train']['seed'])

        # ARGUMENTS: PLOTS DIRECTORY
        self.parser.add_argument('--plots', type=str, help='Path to the plots.json configuration file.')

        # ARGUMENTS: DIRECTORIES
        self.parser.add_argument('--log_dir', type=str, default=self.config['paths']['log_dir'], help='Path where logs will be saved.')

        self.parser.add_argument('--checkpoint_dir', type=str, default=self.config['paths']['checkpoint_dir'], help='Path where models will be saved.')

        self.parser.add_argument('--gen_dir', type=str, default=self.config['paths']['gen_dir'], help='Path where generated images will be saved.')

        self.parser.add_argument('--gen_test_dir', type=str, default=self.config['paths']['gen_test_dir'], help='Path where generated test images will be saved.')

        # ARGUMENTS: DATASET
        self.parser.add_argument('--dataset_train', type=str, default=self.config['dataset']['dataset_train'], help='Path to the pre-processed dataset for training.')

        self.parser.add_argument('--dataset_test', type=str, default=self.config['dataset']['dataset_test'], help='Path to the pre-processed dataset for testing.')

        self.parser.add_argument('--csv_train', type=str, default=self.config['dataset']['csv_train'], help='Path to CSV file needed for torch.utils.data.Dataset to load data for training.')

        self.parser.add_argument('--csv_test', type=str, default=self.config['dataset']['csv_test'], help='Path to CSV file needed for torch.utils.data.Dataset to load data for testing.')

        self.parser.add_argument('--image_size', default=self.config['dataset']['image_size'], type=int, help='Image size')

        self.parser.add_argument('--channels', default=self.config['dataset']['channels'], type=int, help='Image channels')

        self.parser.add_argument('--normalize', nargs='+', default=self.config['dataset']['normalize'], type=float, help='Image normalization: mean, std')

        self.parser.add_argument('--iterations', default=self.config['train']['iterations'], type=int, help='Limit iteration per epoch; 0: no limit, >0: limit')

        self.parser.add_argument('--shuffle', action='store_false' if self.config['dataset']['shuffle'] else 'store_true')

        self.parser.add_argument('--rotation_angle', type=int, default=self.config['dataset']['augmentation']['rotation_angle'], help='Angle for random image rotation when loading data.')

        self.parser.add_argument('--horizontal_flip', action='store_false' if self.config['dataset']['augmentation']['horizontal_flip'] else 'store_true', help='Random horizontal flip when loading data.')

        # ARGUMENTS: HYPERPARAMETERS
        self.parser.add_argument('--batch_size', default=self.config['train']['batch_size'], type=int, help='Batch size')

        self.parser.add_argument('--epochs', default=self.config['train']['epochs'], type=int, help='Epochs to train')

        self.parser.add_argument('--grad_clip', default=self.config['train']['grad_clip'], type=float, help='Use gradient clipping')

        # ARGUMENTS: OPTIMIZER
        self.parser.add_argument('--overwrite_optim', action='store_false' if self.config['train']['optimizer']['overwrite_optim'] else 'store_true', help='If flag is set, and training is continued from checkpoint, then the optimizer settings will be overwritten.')

        self.parser.add_argument('--beta1', default=self.config['train']['optimizer']['beta1'], type=float, help='Beta1 of Adam optimizer')

        self.parser.add_argument('--beta2', default=self.config['train']['optimizer']['beta2'], type=float, help='Beta2 of Adam optimizer')

        self.parser.add_argument('--weight_decay', default=self.config['train']['optimizer']['weight_decay'], type=float, help='Weight decay of optimizer')

        # ARGUMENTS: SCHEDULER
        if 'lr_step_decay' in self.config['train']['optimizer']:
            self.parser.add_argument('--step_size', default=self.config['train']['optimizer']['lr_step_decay']['step_size'], type=int, help='Schedule to decrease learning rate every step_size.')

            self.parser.add_argument('--gamma', default=self.config['train']['optimizer']['lr_step_decay']['gamma'], type=float, help='Decrease learning rate by lr = lr * gamma')

        if 'lr_linear_decay' in self.config['train']['optimizer']:
            self.parser.add_argument('--epoch_range', nargs='+', default=self.config['train']['optimizer']['lr_linear_decay']['epoch_range'], type=int, help='Schedule to decrease learning rate from epoch_start to epoch_end.')

            if self.method == Method.CREATION:
                self.parser.add_argument('--lr_g_end', default=self.config['train']['optimizer']['lr_linear_decay']['lr_g_end'], type=float, help='Last learning rate of generator. If lr_end is reached, the learning rate will no longer be changed.')
                self.parser.add_argument('--lr_d_end', default=self.config['train']['optimizer']['lr_linear_decay']['lr_d_end'], type=float, help='Last learning rate of discriminator. If lr_end is reached, the learning rate will no longer be changed.')

            elif self.method == Method.DETECTION:
                self.parser.add_argument('--lr_end', default=self.config['train']['optimizer']['lr_linear_decay']['lr_end'], type=float, help='Last learning rate. If lr_end is reached, the learning rate will no longer be changed.')

        if 'lr_plateau_decay' in self.config['train']['optimizer']:
            self.parser.add_argument('--plateau_mode', type=str, default=self.config['train']['optimizer']['lr_plateau_decay']['plateau_mode'], help='mode: min | max')
            self.check_error(self.config['train']['optimizer']['lr_plateau_decay'], 'plateau_mode', ['min', 'max'])

            self.parser.add_argument('--plateau_factor', default=self.config['train']['optimizer']['lr_plateau_decay']['plateau_factor'], type=float)

            self.parser.add_argument('--plateau_patience', default=self.config['train']['optimizer']['lr_plateau_decay']['plateau_patience'], type=int)

            if self.method == Method.CREATION:
                self.parser.add_argument('--plateau_min_lr_g', default=self.config['train']['optimizer']['lr_plateau_decay']['plateau_min_lr_g'], type=float)
                self.parser.add_argument('--plateau_min_lr_d', default=self.config['train']['optimizer']['lr_plateau_decay']['plateau_min_lr_d'], type=float)
            
            elif self.method == Method.DETECTION:
                self.parser.add_argument('--plateau_min_lr', default=self.config['train']['optimizer']['lr_plateau_decay']['plateau_min_lr'], type=float)

        if 'lr_cyclic_decay' in self.config['train']['optimizer']:
            if self.method == Method.DETECTION:
                self.parser.add_argument('--lr_max', default=self.config['train']['optimizer']['lr_cyclic_decay']['lr_max'], type=float)

        ##### CREATION #####
        if self.method == Method.CREATION:
            # ARGUMENTS: DATASET
            self.parser.add_argument('--landmark_type', type=str, default=self.config['train']['landmark_type'], help='Facial landmark type: boundary | keypoint')
            self.check_error(self.config['train'], 'landmark_type', ['boundary', 'keypoint'])

            self.parser.add_argument('--vgg_type', type=str, default=self.config['train']['vgg_type'], help='Perceptual network: vgg16 | vggface')
            self.check_error(self.config['train'], 'vgg_type', ['vgg16', 'vggface'])

            self.parser.add_argument('--spec_norm', action='store_false' if self.config['train']['spec_norm'] else 'store_true')

            # ARGUMENTS: HYPERPARAMETERS
            self.parser.add_argument('--lr_g', default=self.config['train']['optimizer']['lr_g'], type=float, help='Learning rate of generator')

            self.parser.add_argument('--lr_d', default=self.config['train']['optimizer']['lr_d'], type=float, help='Learning rate of discriminator')

            self.parser.add_argument('--d_iters', default=self.config['train']['update_strategy']['d_iters'], type=int, help='Fixed update interval of discriminator')

            self.parser.add_argument('--loss_coeff', default=self.config['train']['update_strategy']['loss_coeff'], type=int, help='Adaptive update interval of discriminator')

            self.parser.add_argument('--conv_blocks_d', default=self.config['train']['conv_blocks_d'], type=int, help='Number of convolutional layers in discriminator: 4 | 6')

            # ARGUMENTS: LOSS WEIGHT
            self.parser.add_argument('--l_adv', default=self.config['train']['loss_weights']['l_adv'], type=float, help='Adversarial loss')

            self.parser.add_argument('--l_rec', default=self.config['train']['loss_weights']['l_rec'], type=float, help='Reconstruction loss')

            self.parser.add_argument('--l_self', default=self.config['train']['loss_weights']['l_self'], type=float, help='Cycle consistency loss')

            self.parser.add_argument('--l_triple', default=self.config['train']['loss_weights']['l_triple'], type=float, help='Triple consistency loss')

            self.parser.add_argument('--l_id', default=self.config['train']['loss_weights']['l_id'], type=float, help='Identity loss')

            self.parser.add_argument('--l_percep', default=self.config['train']['loss_weights']['l_percep'], type=float, help='Perceptual loss')

            self.parser.add_argument('--l_fm', default=self.config['train']['loss_weights']['l_fm'], type=float, help='Feature Matching loss')

            self.parser.add_argument('--l_tv', default=self.config['train']['loss_weights']['l_tv'], type=float, help='Total variation loss')

            self.parser.add_argument('--l_gp', default=self.config['train']['loss_weights']['l_gp'], type=float, help='Gradient penalty loss')

            self.parser.add_argument('--l_gc', default=self.config['train']['loss_weights']['l_gc'], type=float, help='Gradient clipping value')

        ##### DETECTION #####
        if self.method == Method.DETECTION:
            self.parser.add_argument('--threshold', default=self.config['train']['threshold'], type=float, help='Threshold for binary classification.')
            # ARGUMENTS: HYPERPARAMETERS
            self.parser.add_argument('--batch_size_class', default=self.config['train']['batch_size_class'], type=int, help='Batch size')

            self.parser.add_argument('--lr', default=self.config['train']['optimizer']['lr'], type=float, help='Learning rate')

            self.parser.add_argument('--margin', default=self.config['train']['margin'], type=float, help='Threshold m for contrastive loss.')

            self.parser.add_argument('--epochs_feature', default=self.config['train']['epochs_feature'], type=int, help='Number of epochs to train features.')

            self.parser.add_argument('--len_feature', default=self.config['train']['len_feature'], type=int, help='Length of feature vector.')

            self.parser.add_argument('--hidden_layer_num_features', default=self.config['train']['hidden_layer_num_features'], type=int, help='Length of hidden layer of classifier.')

            self.parser.add_argument('--l_mask', default=self.config['train']['loss_weights']['l_mask'], type=float, help='Mask loss.')

            self.parser.add_argument('--l_mask_sv', default=self.config['train']['loss_weights']['l_mask_sv'], type=float, help='Mask loss regression type: 1=supervised | 2=unsupervised')

            self.parser.add_argument('--loss_type', type=str, default=self.config['train']['loss_type'], help='Loss type for feature extraction: contrastive | triplet')
            self.check_error(self.config['train'], 'loss_type', ['contrastive', 'triplet'])

            # ARGUMENTS: DATASET
            self.parser.add_argument('--mask_size', default=self.config['dataset']['mask_size'], type=int, help='Mask size')
