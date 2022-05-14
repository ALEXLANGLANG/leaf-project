import shutil

import torch


def load_checkpoint(model_path,device):
    weights = torch.load(model_path,map_location=torch.device(device))
    epoch = None
    if 'epoch' in weights:
        epoch = weights.pop('epoch')
    if 'state_dict' in weights:
        state_dict = (weights['state_dict'])
    else:
        state_dict = weights
    return epoch, state_dict


def weight_parameters(module):
    return [param for name, param in module.named_parameters() if 'weight' in name]


def bias_parameters(module):
    return [param for name, param in module.named_parameters() if 'bias' in name]


def save_checkpoint(save_path, states, file_prefixes, is_best, filename='ckpt.pth.tar'):
    def run_one_sample(save_path_, state_, prefix_, is_best_, filename_):
        torch.save(state_, save_path_ / f'{prefix_}_{filename_}')
        if is_best_:
            shutil.copyfile(save_path_ / f'{prefix_}_{filename_}',
                            save_path_ / f'{prefix_}_model_best.pth.tar')

    if not isinstance(file_prefixes, str):
        for (prefix, state) in zip(file_prefixes, states):
            run_one_sample(save_path, state, prefix, is_best, filename)

    else:
        run_one_sample(save_path, states, file_prefixes, is_best, filename)
