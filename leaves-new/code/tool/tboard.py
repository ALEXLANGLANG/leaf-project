from torch import TensorType
import numpy as np
import time
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.plugins.hparams import plugin_data_pb2
from os import walk
from os import path as osp
import re


def now_file_name():
    t = time.localtime(time.time())
    fmt = '{}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}'
    name = fmt.format(t.tm_year, t.tm_mon, t.tm_mday,
                      t.tm_hour, t.tm_min, t.tm_sec)
    return name


def to_scalar(value):
    if type(value) == TensorType:
        value = value.numpy()
    if type(value) == np.ndarray:
        shape = value.shape
        fmt = 'found an array or tensor of shape {} where a scalar is expected'
        assert len(shape) == 0 or shape == (1,), fmt.format(shape)
        value = value.item()
    return value


def write_data_dict(data, timestamp, writer, prefix=''):
    usage = 'dictionary leaves must be either scalars (for single plot)' +\
            ' or lists of single-item {label: scalar} dictionaries (for multiple plots with legend)'
    for postfix, value in data.items():
        tag = '/'.join((prefix, str(postfix))) if prefix != '' else str(postfix)
        if type(value) == dict:
            write_data_dict(value, timestamp, writer, tag)
        else:
            if type(value) == list or type(value) == tuple:
                assert all([type(item) == dict and len(item) == 1 for item in value]), usage
                if len(value):
                    scalars = {}
                    for item in value:
                        label, v = next(iter(item.items()))
                        v = to_scalar(v)
                        if v is not None:
                            scalars[label] = to_scalar(v)
                    if len(scalars):
                        writer.add_scalars(tag, scalars, timestamp)
            else:
                scalar = to_scalar(value)
                if scalar is not None:
                    # print('write_data_dict value:', scalar)
                    writer.add_scalar(tag, scalar, timestamp)


def read_tensorboard_events(acc, tag):
    events = [(event.step, event.value) for event in acc.Scalars(tag)]
    return zip(*events)


def read_tensorboard_file(file_name):
    print('reading {}'.format(file_name), end='')
    acc = event_accumulator.EventAccumulator(file_name,
                                             size_guidance={'scalars': 0})
    acc.Reload()
    print(' done.')
    return acc


def read_tensorboard_event_file(file_name, tags=None):
    acc = read_tensorboard_file(file_name)

    if tags is None:
        tags = acc.Tags()['scalars']
    data = []
    for tag in tags:
        x, y = read_tensorboard_events(acc, tag)
        # print(tag, y)
        data.append({'tag': tag, 'x': x, 'y': y})
    return data


# Not sure why the parameters are described but their values do not
# show up in parameters. So for now we leave this function unused.
# The plugin_data_pb2 interface is not officially supported anyway
def read_tensorboard_hparam_file(file_name):
    acc = read_tensorboard_file(file_name)
    raw = acc.summary_metadata['_hparams_/experiment'].plugin_data
    parameters = plugin_data_pb2.HParamsPluginData.FromString(raw.content)
    return parameters


def read_tensorboard_run(run_name, log_dir='runs', experiment_names=None, tags=None, skip_hyper=True):
    assert skip_hyper, 'hyper-parameter loading is not yet implemented'
    dir_name = osp.join(log_dir, run_name)
    assert osp.exists(dir_name), \
        'directory {} does not exist'.format(dir_name)
    digits_and_dot = re.compile(r'^\d*.\d*$')
    data = []
    for path, _, base_names in walk(dir_name):
        for base_name in base_names:
            if 'events.out.tfevents' in base_name:
                experiment_name = osp.basename(path)
                if skip_hyper:
                    if digits_and_dot.match(experiment_name) is not None:
                        continue
                if experiment_names is not None:
                    if experiment_name not in experiment_names:
                        continue
                file_name = osp.join(path, base_name)
                events = read_tensorboard_event_file(file_name, tags=tags)
                for event in events:
                    event['experiment'] = experiment_name
                data.extend(events)
    return data



