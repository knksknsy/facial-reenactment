import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, bias=False, instance_norm=True, activation='relu', deconv=False):
        super(ConvBlock, self).__init__()
        self.instance_norm = instance_norm
        self.deconv = deconv
        self.activation = activation

        if padding is None:
            padding = kernel_size // 2

        if not self.deconv:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        else:
            self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if self.instance_norm:
            self.in2d = nn.InstanceNorm2d(num_features=out_channels, affine=True, track_running_stats=True)

        if self.activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self.activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.01)
        elif self.activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x):
        out = self.conv2d(x)

        if self.instance_norm:
            out = self.in2d(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        super(DownSamplingBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, bias)


    def forward(self, x):
        return self.conv_block(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        super(UpSamplingBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, bias, deconv=True)


    def forward(self, x):
        return self.conv_block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        layers = []

        layers.append(ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=None))

        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return x + self.layers(x)
