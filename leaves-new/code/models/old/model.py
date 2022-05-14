from keras import backend as kb
from keras import initializers
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dropout, BatchNormalization, Lambda, LeakyReLU
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.utils import multi_gpu_model
import math
import string
from yaml_io import namespace_to_dict
from hardware import get_number_of_cores, get_gpus
from loss import get_losses
import json

letters = string.ascii_lowercase


def encoder_decoder(config, is_training_time):

    print('Encoder-decoder parameters:')
    for key, value in namespace_to_dict(config).items():
        print('{}: {}'.format(key, value))

    pyramid = image_pyramid(config.tile_height, config.tile_width, config.input_channels, config.output_scales)

    embeddings = encoder(pyramid, block_layers=config.block_layers,
                         embedding_dimension=config.pyramid_feature_dimension, name='1')

    outputs, names = decoder(pyramid, embeddings, block_layers=config.block_layers,
                             internal_channels=config.decoder_channels,
                             use_batch_normalization=config.batch_normalization,
                             use_dropout=config.dropout, is_training_time=is_training_time)

    net = Model(inputs=pyramid, outputs=outputs)
    return net, names


def image_pyramid(height, width, channels, levels):
    pyramid = []
    for level in range(levels):
        # TODO: stop if images get too small
        name = 'img1' if level == 0 else 'img1_' + letters[level - 1]
        pyramid.append(Input(shape=(height, width, channels), name=name))
        height, width = math.ceil(height/2), math.ceil(width/2)
    return pyramid


def encoder(pyramid, block_layers=3, embedding_dimension=16, name=''):

    def block(image, channels, level_id, number_of_layers):
        y = image
        for layer in range(number_of_layers):
            stride = 2 if layer == number_of_layers - 1 else 1
            suffix = '{}_{}{}'.format(level_id, letters[layer], name)
            y = Conv2D(channels, (3, 3), strides=stride, padding='same', name='conv' + suffix)(y)
            y = BatchNormalization(center=False, scale=False, name='bn' + suffix)(y)
            y = LeakyReLU(alpha=0.1, name='lrelu' + suffix)(y)
        f = Conv2D(embedding_dimension, (1, 1), padding='same', activation='relu',
                   name='out_f{}{}'.format(level_id, name))(y)
        return y, f

    pyramid_levels = len(pyramid)
    embeddings, dimension = [], embedding_dimension
    activation = pyramid[0]
    for level in range(1, pyramid_levels):
        activation, feature = block(activation, dimension, level, block_layers)
        embeddings.append(feature)
        if level < pyramid_levels - 1:
            activation = Lambda(concatenate)([activation, pyramid[level]])
        dimension *= 2

    return embeddings


def decoder(pyramid, embeddings, block_layers=3, internal_channels=64,
            use_batch_normalization=1, use_dropout=1, is_training_time=1):
    def block(y, pyramid_level, number_of_layers, channels, branch_channels):
        def decoding_layer(z, number_of_channels, block_id, layer_number):
            layer_id = layer_number + 1
            z = Conv2D(number_of_channels, (3, 3), padding='same',
                       name='block{}_conv{}'.format(block_id, layer_id))(z)
            if layer_number == 0 and use_batch_normalization == 1:
                z = BatchNormalization(name='block{}_bn'.format(block_id))(z)
            z = LeakyReLU(alpha=0.1, name='lrelu{}_{}'.format(block_id, layer_id))(z)
            if layer_number == 0 and use_dropout == 1 and is_training_time == 1:
                z = Dropout(rate=0.1, name='block{}_drop{}'.format(block_id, layer_id))(z)
            return z

        block_number = pyramid_levels - pyramid_level
        for layer in range(number_of_layers):
            y = decoding_layer(y, channels, block_number, layer)
            if layer % 2 == 1:
                channels //= 2

        est, act, act_name = None, None, None
        if branch_channels > 1:
            est = side_branch(y, branch_channels, 'block{}'.format(block_number))
            act_name = 'o{}'.format(block_number)
            act = Activation('sigmoid', name=act_name)(est)

        if block_number > 1:
            y = Conv2D(64, (3, 3), strides=2, padding='same', name='block{}_conv_pool'.format(block_number))(y)
            y = LeakyReLU(alpha=0.1, name='block{}_lrelu_pool'.format(block_number))(y)

        return y, est, act, act_name

    pyramid_levels = len(pyramid)
    estimates, activations, names, x = [], [], [], []
    layers, start_channels, side_channels = block_layers, internal_channels, 1
    for level in range(pyramid_levels):
        x = pyramid[level] if level == 0 else \
            Lambda(concatenate)([x, pyramid[level], embeddings[level - 1]])
        x, estimate, activation, name = block(x, level, layers, start_channels, side_channels)
        if name is not None:
            names.append(name)
        if estimate is not None:
            estimates.insert(0, estimate)
            activations.insert(0, activation)
        side_channels *= 2

    # fuse
    seg_fuse = Lambda(concatenate)(estimates)
    est_fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None,
                      kernel_initializer=initializers.Constant(value=1/4),
                      name='segfuse_conv')(seg_fuse)
    name = 'ofuse'
    activations.append(Activation('sigmoid', name=name)(est_fuse))
    names.append(name)

    return activations, names


def concatenate(xs):
    return xs[0] if len(xs) == 1 else kb.concatenate(xs, -1)


def side_branch(x, factor, name):
    x = Conv2D(1, (1, 1), activation=None, padding='same', name=name + '5')(x)
    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same',
                        use_bias=False, activation=None, name=name + '6')(x)
    return x


def load_model(conf, input_channels, checkpoint_file=None, is_training_time=True):

    print('Loading Network...')
    m = conf.model
    m.input_channels = input_channels
    net1, activation_names = encoder_decoder(m, is_training_time=is_training_time)
    net1.summary()

    t = conf.experiment.training

    workers = get_number_of_cores()
    if workers is None:
        workers = t.default_workers
        msg = 'Could not get the number of cores. Using {} worker{} by default'
        print(msg.format(workers, '' if workers == 1 else 's'))
    else:
        if workers == 0:
            workers = 1
        print('Using {} worker{}'.format(workers, '' if workers == 1 else 's'))

    if t.num_gpus > 0:
        gpus = get_gpus()
        gpus_found = len(gpus)
        if gpus_found < t.num_gpus:
            msg = '{} GPU{} requested but only {} found. Downgrading request.'
            print(msg.format(t.num_gpus, '' if t.num_gpus == 1 else 's', gpus_found))
            t.num_gpus = gpus_found

    if t.num_gpus > 0:
        print('Using {} GPU{}'.format(t.num_gpus, '' if t.num_gpus == 1 else 's'))
        if t.num_gpus > 1:
            net = multi_gpu_model(net1, gpus=t.num_gpus)
        else:
            net = net1
    else:
        print('Using CPU')
        net = net1

    if checkpoint_file is not None:
        print('Loading network weights from {}'.format(checkpoint_file), end='')
        net.load_weights(checkpoint_file, by_name=True)
        print('. Done.')

    losses = get_losses(m.tile_margin,
                        t.focal_loss_alpha, t.focal_loss_gamma, activation_names)

    if t.optimizer == 'adam':
        optimizer = optimizers.Adam(lr=t.learning_rate)
    elif t.optimizer == 'sgd':
        optimizer = optimizers.SGD(lr=t.learning_rate, decay=t.decay, momentum=t.momentum)
    else:
        raise ValueError('Unknown optimizer type {}'.format(t.optimizer))

    net.compile(loss=losses, metrics=["accuracy"], optimizer=optimizer)

    return net, workers, activation_names


def save_performance(names, scores, n_samples, file_name):
    info = {name: score for name, score in zip(names, scores)}
    info['number of inputs'] = n_samples
    with open(file_name, 'w') as file:
        json.dump(info, file)
    print('Saved performance scores to {}'.format(file_name))
