from configs import Options

class TrainOptions(Options):
    def __init__(self, description):
        super(TrainOptions, self).__init__(description)
        self._init_parser()
        self._parse_args()
        self.losses = self._parse_losses(self.args)


    def _init_parser(self):
        # ARGUMENTS: OPTIONS
        self.parser.add_argument('--device', nargs='?', default=self.config['device'], const=self.config['device'],
                                            choices=self.config['device_options'],
                                            help='Whether to run the model on GPU or CPU.')
        self.parser.add_argument('--pin_memory', action='store_false' if self.config['pin_memory'] else 'store_true')
        self.parser.add_argument('--num_workers', type=int, default=self.config['num_workers'])
        self.parser.add_argument('--continue_id', type=str, default=self.config['train']['continue_id'],
                                            help='Id of the models to continue training.')
        self.parser.add_argument('--checkpoint_freq', type=int, default=self.config['train']['checkpoint_freq'],
                                            help='Frequency in which model checkpoints will be saved')
        self.parser.add_argument('--log_freq', type=int, default=self.config['train']['log_freq'],
                                            help='Frequency in which logs will be saved')
        self.parser.add_argument('--test', action='store_false' if self.config['train']['test'] else 'store_true',
                                            help='Model will be tested after each epoch.')
        self.parser.add_argument('--test_train', action='store_false' if self.config['train']['test_train'] else 'store_true',
                                            help='Evaluations will be calculated for train set during training.')
        self.parser.add_argument('--seed', type=int, default=self.config['train']['seed'])

        # ARGUMENTS: DIRECTORIES
        self.parser.add_argument('--log_dir', type=str, default=self.config['paths']['log_dir'],
                                            help='Path where logs will be saved.')
        self.parser.add_argument('--checkpoint_dir', type=str, default=self.config['paths']['checkpoint_dir'],
                                            help='Path where models will be saved.')
        self.parser.add_argument('--gen_dir', type=str, default=self.config['paths']['gen_dir'],
                                            help='Path where generated images will be saved.')
        self.parser.add_argument('--gen_test_dir', type=str, default=self.config['paths']['gen_test_dir'],
                                            help='Path where generated test images will be saved.')

        # ARGUMENTS: DATASET
        self.parser.add_argument('--dataset_train', type=str, default=self.config['dataset']['dataset_train'],
                                            help='Path to the pre-processed dataset for training.')
        self.parser.add_argument('--dataset_test', type=str, default=self.config['dataset']['dataset_test'],
                                            help='Path to the pre-processed dataset for testing.')
        self.parser.add_argument('--csv_train', type=str, default=self.config['dataset']['csv_train'],
                                            help='Path to CSV file needed for torch.utils.data.Dataset to load data for training.')
        self.parser.add_argument('--csv_test', type=str, default=self.config['dataset']['csv_test'],
                                            help='Path to CSV file needed for torch.utils.data.Dataset to load data for testing.')
        self.parser.add_argument('--image_size', default=self.config['dataset']['image_size'], type=int,
                                            help='Image size')
        self.parser.add_argument('--channels', default=self.config['dataset']['channels'], type=int,
                                            help='Image channels')
        self.parser.add_argument('--normalize', nargs='+', default=self.config['dataset']['normalize'], type=float,
                                            help='Image normalization: mean, std')
        self.parser.add_argument('--iterations', default=self.config['train']['iterations'], type=int,
                                            help='Limit iteration per epoch; 0: no limit, >0: limit')
        self.parser.add_argument('--landmark_type', type=str, default=self.config['train']['landmark_type'],
                                            help='Facial landmark type: boundary | keypoint')
        self.parser.add_argument('--vgg_type', type=str, default=self.config['train']['vgg_type'],
                                            help='Perceptual network: vgg16 | vggface')
        self.parser.add_argument('--spec_norm', action='store_false' if self.config['train']['spec_norm'] else 'store_true')
        self.parser.add_argument('--shuffle', action='store_false' if self.config['dataset']['shuffle'] else 'store_true')
        self.parser.add_argument('--rotation_angle', type=int, default=self.config['dataset']['augmentation']['rotation_angle'],
                                            help='Angle for random image rotation when loading data.')
        self.parser.add_argument('--horizontal_flip', action='store_false' if self.config['dataset']['augmentation']['horizontal_flip'] else 'store_true',
                                            help='Random horizontal flip when loading data.')

        # ARGUMENTS: LOSS WEIGHT
        self.parser.add_argument('--l_adv', default=self.config['train']['loss_weights']['l_adv'], type=float,
                                            help='Adversarial loss')
        self.parser.add_argument('--l_rec', default=self.config['train']['loss_weights']['l_rec'], type=float,
                                            help='Reconstruction loss')
        self.parser.add_argument('--l_self', default=self.config['train']['loss_weights']['l_self'], type=float,
                                            help='Cycle consistency loss')
        self.parser.add_argument('--l_triple', default=self.config['train']['loss_weights']['l_triple'], type=float,
                                            help='Triple consistency loss')
        self.parser.add_argument('--l_id', default=self.config['train']['loss_weights']['l_id'], type=float,
                                            help='Identity loss')
        self.parser.add_argument('--l_percep', default=self.config['train']['loss_weights']['l_percep'], type=float,
                                            help='Perceptual loss')
        self.parser.add_argument('--l_feature_matching', default=self.config['train']['loss_weights']['l_feature_matching'], type=float,
                                            help='Feature Matching loss')
        self.parser.add_argument('--l_tv', default=self.config['train']['loss_weights']['l_tv'], type=float,
                                            help='Total variation loss')
        self.parser.add_argument('--l_gp', default=self.config['train']['loss_weights']['l_gp'], type=float,
                                            help='Gradient penalty loss')
        self.parser.add_argument('--l_gc', default=self.config['train']['loss_weights']['l_gc'], type=float,
                                            help='Gradient clipping value')

        # ARGUMENTS: HYPERPARAMETERS
        self.parser.add_argument('--batch_size', default=self.config['train']['batch_size'], type=int,
                                            help='Batch size')
        self.parser.add_argument('--epochs', default=self.config['train']['epochs'], type=int,
                                            help='Epochs to train')
        self.parser.add_argument('--lr_g', default=self.config['train']['optimizer']['lr_g'], type=float,
                                            help='Learning rate of generator')
        self.parser.add_argument('--lr_d', default=self.config['train']['optimizer']['lr_d'], type=float,
                                            help='Learning rate of discriminator')
        self.parser.add_argument('--grad_clip', default=self.config['train']['grad_clip'], type=float,
                                            help='Use gradient clipping')
        self.parser.add_argument('--d_iters', default=self.config['train']['update_strategy']['d_iters'], type=int,
                                            help='Fixed update interval of discriminator')
        self.parser.add_argument('--loss_coeff', default=self.config['train']['update_strategy']['loss_coeff'], type=int,
                                            help='Adaptive update interval of discriminator')

        # ARGUMENTS: OPTIMIZER
        self.parser.add_argument('--overwrite_optim', action='store_false' if self.config['train']['optimizer']['overwrite_optim'] else 'store_true',
                                            help='If flag is set, and training is continued from checkpoint, then the optimizer settings will be overwritten.')
        self.parser.add_argument('--beta1', default=self.config['train']['optimizer']['beta1'], type=float,
                                            help='Beta1 of Adam optimizer')
        self.parser.add_argument('--beta2', default=self.config['train']['optimizer']['beta2'], type=float,
                                            help='Beta2 of Adam optimizer')
        self.parser.add_argument('--weight_decay', default=self.config['train']['optimizer']['weight_decay'], type=float,
                                            help='Weight decay of optimizer')
        if 'lr_linear_decay' in self.config['train']['optimizer']:
            self.parser.add_argument('--epoch_range', nargs='+', default=self.config['train']['optimizer']['lr_linear_decay']['epoch_range'], type=int,
                                                help='Schedule to decrease learning rate from epoch_start to epoch_end.')
            self.parser.add_argument('--lr_g_end', default=self.config['train']['optimizer']['lr_linear_decay']['lr_g_end'], type=float,
                                                help='Last learning rate of generator. If lr_end is reached, the learning rate will no longer be changed.')
            self.parser.add_argument('--lr_d_end', default=self.config['train']['optimizer']['lr_linear_decay']['lr_d_end'], type=float,
                                                help='Last learning rate of discriminator. If lr_end is reached, the learning rate will no longer be changed.')
        if 'lr_step_decay' in self.config['train']['optimizer']:
            self.parser.add_argument('--step_size', default=self.config['train']['optimizer']['lr_step_decay']['step_size'], type=int,
                                                help='Schedule to decrease learning rate every step_size.')
            self.parser.add_argument('--gamma', default=self.config['train']['optimizer']['lr_step_decay']['gamma'], type=float,
                                                help='Decrease learning rate by lr = lr * gamma')


    def _parse_losses(self, args):
        loss_keys = [k for k in list(args.__dict__.keys()) if 'l_' in k]
        losses = dict(zip([k.replace('l_', '') for k in loss_keys], [float(args.__getattribute__(v)) for v in loss_keys]))
        return losses
