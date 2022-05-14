# Pytorch ENet implementation adapted from https://github.com/gjy3035/enet.pytorch by Junyu Gao
# See https://arxiv.org/abs/1606.02147 for a description of ENet by Paszke et al

import torch
import torch.nn as nn
import torch.nn.functional as fct
from torch.autograd import Variable


class InitialBlock(nn.Module):
    def __init__(self, input_channels, branch_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, branch_channels, (3, 3), stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(branch_channels, 1e-3)
        self.prelu = nn.PReLU(branch_channels)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        y = torch.cat([
            self.prelu(self.batch_norm(self.conv(x))), self.pool(x)
        ], 1)
        return y


def _prelu(channels, use_relu):
    return nn.PReLU(channels) if use_relu is False else nn.ReLU()


class Bottleneck(nn.Module):
    """
    The bottleneck module has three different variants:
    1. A regular convolution which you can decide whether or not to downsample.
    2. A dilated convolution which requires you to have a dilation factor.
    3. A separable convolution that separates the kernel into a 5x1 and a 1x5 kernel.
    INPUTS:
    - inputs(Tensor): a 4D Tensor of the previous convolutional block of shape
    [batch_size, channel, height, width].
    - output_channels(int): an integer indicating the output depth of the
    output convolutional block.
    - regularizer_prob(float): the float p that represents the prob of
    dropping a layer for spatial dropout regularization.
    - downsampling(bool): if True, a max-pool2D layer is added to downsample
    the spatial sizes.
    - upsampling(bool): if True, the upsampling bottleneck is activated but
    requires pooling indices to upsample.
    - dilated(bool): if True, then dilated convolution is done, but requires
    a dilation rate to be given.
    - dilation_rate(int): the dilation factor for performing atrous
    convolution/dilated convolution
    - separable(bool): if True, then separable convolution is done, and
    the only filter size used here is 5.
    - use_relu(bool): if True, then all the prelus become relus.
    """

    def __init__(self,
                 input_channels=None,
                 output_channels=None,
                 regularizer_prob=0.1,
                 downsampling=False,
                 upsampling=False,
                 dilated=False,
                 dilation_rate=None,
                 separable=False,
                 use_relu=False):
        super(Bottleneck, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.downsampling = downsampling
        self.upsampling = upsampling
        self.use_relu = use_relu

        internal = output_channels // 4
        input_stride = 2 if downsampling else 1
        # First projection with 1x1 kernel (2x2 for down-sampling)
        conv1x1_1 = nn.Conv2d(input_channels, internal,
                              input_stride, input_stride, bias=False)
        batch_norm1 = nn.BatchNorm2d(internal, 1e-3)
        prelu1 = _prelu(internal, use_relu)
        self.block1x1_1 = nn.Sequential(conv1x1_1, batch_norm1, prelu1)

        if downsampling:
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
            conv = nn.Conv2d(internal, internal, 3, stride=1, padding=1)
        elif upsampling:
            # padding is replaced with spatial convolution without bias.
            spatial_conv = nn.Conv2d(input_channels, output_channels, 1,
                                     bias=False)
            batch_norm = nn.BatchNorm2d(output_channels, 1e-3)
            self.conv_before_unpool = nn.Sequential(spatial_conv, batch_norm)
            self.unpool = nn.MaxUnpool2d(2)
            conv = nn.ConvTranspose2d(internal, internal, 3,
                                      stride=2, padding=1, output_padding=1)
        elif dilated:
            conv = nn.Conv2d(internal, internal, 3, padding=dilation_rate,
                             dilation=dilation_rate)
        elif separable:
            conv1 = nn.Conv2d(internal, internal, (5, 1), padding=(2, 0),
                              bias=False)
            conv2 = nn.Conv2d(internal, internal, (1, 5), padding=(0, 2))
            conv = nn.Sequential(conv1, conv2)
        else:
            conv = nn.Conv2d(internal, internal, 3, padding=1)

        batch_norm = nn.BatchNorm2d(internal, 1e-3)
        prelu = _prelu(internal, use_relu)
        self.middle_block = nn.Sequential(conv, batch_norm, prelu)

        # Final projection with 1x1 kernel
        conv1x1_2 = nn.Conv2d(internal, output_channels, 1, bias=False)
        batch_norm2 = nn.BatchNorm2d(output_channels, 1e-3)
        prelu2 = _prelu(output_channels, use_relu)
        self.block1x1_2 = nn.Sequential(conv1x1_2, batch_norm2, prelu2)

        # regularize
        self.dropout = nn.Dropout2d(regularizer_prob)

    def forward(self, x, pooling_indices=None):
        input_shape = x.size()
        indices = None
        if self.downsampling:
            main, indices = self.pool(x)
            if self.output_channels != self.input_channels:
                pad = Variable(torch.Tensor(input_shape[0],
                                            self.output_channels - self.input_channels,
                                            input_shape[2] // 2,
                                            input_shape[3] // 2).zero_(), requires_grad=False)
                # if (torch.cuda.is_available):
                #     pad = pad.cuda(0)
                main = torch.cat((main, pad), 1)
        elif self.upsampling:
            main = self.unpool(self.conv_before_unpool(x), pooling_indices)
        else:
            main = x

        other_net = nn.Sequential(self.block1x1_1, self.middle_block,
                                  self.block1x1_2)
        other = other_net(x)
        y = fct.relu(main + other)
        if self.downsampling:
            return y, indices
        return y


def bottleneck_name_list(stage, n, start=0):
    return ['bottleneck_{}_{}'.format(stage, k) for k in range(start, n)]


ENCODER_LAYER_NAMES = ['initial'] + \
                      bottleneck_name_list(1, 5) + \
                      bottleneck_name_list(2, 9) + \
                      bottleneck_name_list(3, 9, start=1) + \
                      ['classifier']

DECODER_LAYER_NAMES = bottleneck_name_list(4, 3) + \
                      bottleneck_name_list(5, 2) + \
                      ['fullconv']


class Encoder(nn.Module):
    """
    See ENet for inputs. If only_encode is True, the output has n_classes channels.
    Use this option to train and use the encoder as standalone.
    If a decoder follows the encoder, set only_encode to False.
    """
    def __init__(self, n_classes, image_size, hidden_channels, only_encode=True):
        super(Encoder, self).__init__()
        self.n_classes = n_classes
        self.image_size = image_size
        self.only_encode = only_encode
        layers = [InitialBlock(image_size[2], hidden_channels[0]),
                  Bottleneck(hidden_channels[1], hidden_channels[2], regularizer_prob=0.01, downsampling=True)]
        layers.extend(4 * [Bottleneck(hidden_channels[2], hidden_channels[2], regularizer_prob=0.01)])

        # Section 2 and 3
        layers.append(Bottleneck(hidden_channels[2], hidden_channels[3], downsampling=True))
        layers.extend(
            2 * [
                Bottleneck(hidden_channels[3], hidden_channels[3]),
                Bottleneck(hidden_channels[3], hidden_channels[3], dilated=True, dilation_rate=2),
                Bottleneck(hidden_channels[3], hidden_channels[3], separable=True),
                Bottleneck(hidden_channels[3], hidden_channels[3], dilated=True, dilation_rate=4),
                Bottleneck(hidden_channels[3], hidden_channels[3]),
                Bottleneck(hidden_channels[3], hidden_channels[3], dilated=True, dilation_rate=8),
                Bottleneck(hidden_channels[3], hidden_channels[3], separable=True),
                Bottleneck(hidden_channels[3], hidden_channels[3], dilated=True, dilation_rate=16)
            ])

        if only_encode:
            layers.append(nn.Conv2d(hidden_channels[3], self.n_classes, 1))

        for layer, layer_name in zip(layers, ENCODER_LAYER_NAMES):
            super(Encoder, self).__setattr__(layer_name, layer)
        self.layers = layers

    def forward(self, x):
        pooling_stack = []
        y = x
        for layer in self.layers:
            if hasattr(layer, 'downsampling') and layer.downsampling:
                y, pooling_indices = layer(y)
                pooling_stack.append(pooling_indices)
            else:
                y = layer(y)

        if self.only_encode:
            y = fct.upsample(y, self.image_size[:2], None, 'bilinear')

        return y, pooling_stack


class Decoder(nn.Module):
    def __init__(self, n_classes, hidden_channels):
        super(Decoder, self).__init__()
        self.n_classes = n_classes
        layers = [
            # Section 4
            Bottleneck(hidden_channels[3], hidden_channels[2], upsampling=True, use_relu=True),
            Bottleneck(hidden_channels[2], hidden_channels[2], use_relu=True),
            Bottleneck(hidden_channels[2], hidden_channels[2], use_relu=True),

            # Section 5
            Bottleneck(hidden_channels[2], hidden_channels[1], upsampling=True, use_relu=True),
            Bottleneck(hidden_channels[1], hidden_channels[1], use_relu=True),
            nn.ConvTranspose2d(hidden_channels[1], self.n_classes, 2, stride=2)
        ]

        self.layers = nn.ModuleList([layer for layer in layers])

    def forward(self, x, pooling_stack):
        y = x
        for layer in self.layers:
            if hasattr(layer, 'upsampling') and layer.upsampling:
                pooling_indices = pooling_stack.pop()
                y = layer(y, pooling_indices)
            else:
                y = layer(y)
        return y


class ENet(nn.Module):
    """
    INPUTS:
        * n_classes (integer): number of classes
        * image_size (triple of integers): rows, columns, bands of the input image
        * hidden_channels (quadruple of integers):
            0: number of channels in each branch of the initial block
            1, 2, 3: Increasing numbers of channels in the encoder bottleneck blocks.
                The same three values are used in reverse order in the decoder.
    """

    def __init__(self, n_classes, image_size, hidden_channels=(13, 16, 64, 128)):
        super(ENet, self).__init__()
        self.encoder = Encoder(n_classes, image_size, hidden_channels, only_encode=False)
        self.decoder = Decoder(n_classes, hidden_channels)

    def forward(self, x):
        y, pooling_stack = self.encoder(x)
        y = self.decoder(y, pooling_stack)
        return y
