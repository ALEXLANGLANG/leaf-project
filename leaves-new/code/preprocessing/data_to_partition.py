import os
from random import sample

import numpy as np
from tqdm import tqdm
import sys



sys.path.append('../')

from tool.data_io import get_data_paths, read_from_yaml, save_json, try_create_dir, read_json, read_image, get_data_labels
from tool.yaml_io import namespace_to_dict

def get_old_partition(path):
    part = {'training': [],
            'validation': [],
            'testing':[]}
    p = read_json(path)
    for phase in p.keys():
        imgs = p[phase]['images']
        for base_name in imgs:
            img_id = os.path.join(imgs[base_name]['species'], base_name)
            part[phase]+=[img_id]
    return part


def add_paths_to_partition(partition, data_paths=None, tile_paths=None):
    part = {}
    n_path = 0
    n_tile = 0
    if data_paths is not None:
        for phase in partition.keys():
            part[phase] = {'images': {}}
            ids = partition[phase]
            for img_id in ids:
                if img_id not in data_paths.keys():
                    continue
                paths = data_paths[img_id]
                paths['id'] = img_id
                part[phase]['images'][img_id] = paths
                n_path += 1
    if tile_paths is not None:
        for phase in partition.keys():
            if phase not in part.keys():  
                part[phase] ={}
            if 'tiles' not in part[phase].keys():
                part[phase]['tiles'] = {}
            img_ids = partition[phase]
            for tile_id in tile_paths.keys():
                img_id = '_'.join(tile_id.split('_')[:-1])
#                 from IPython import embed; embed()
#                 exit()
                if img_id in img_ids:
                    paths = tile_paths[tile_id]
                    paths['id']=tile_id
                    part[phase]['tiles'][tile_id] = paths
                    n_tile+=1
    print(f'Warning: add_paths_to_partition: add {n_path} paths and {n_tile} tiles')
    return part


def add_tile_ratio_to_partition(root, partition, type_='label'):
    def get_pos_ratio(path):
        label, _ = read_image(path)
        tot = label.shape[0] * label.shape[1]
        pos = (label != 0).sum()
        return pos / tot
    n_tot = sum([len(partition[phase]['tiles'].keys()) for phase in partition.keys()])
    with tqdm(total=n_tot, unit='tile') as pbar:
        for phase in partition.keys():
            for t_id in partition[phase]['tiles'].keys():
                path = os.path.join(root, partition[phase]['tiles'][t_id][type_])
                partition[phase]['tiles'][t_id][f'{type_}_ratio'] = get_pos_ratio(path)
                pbar.update()
    print(f'Warning: add_tile_ratio_to_partition: add pos/tot {type_} ratio to {n_tot} tiles')
    return partition

def add_info_to_partition(root, partition,type_='label'):
    def get_info(path):
        label, _ = read_image(path)
        return {'height':label.shape[0], 'width':label.shape[1]}   
    n_tot = sum([len(partition[phase]['images'].keys()) for phase in partition.keys()])
    with tqdm(total=n_tot, unit='image') as pbar:
        for phase in partition.keys():
            for i_id in partition[phase]['images'].keys():
                path = os.path.join(root, partition[phase]['images'][i_id][type_])
                partition[phase]['images'][i_id]['info'] = get_info(path)
                pbar.update()
    print(f'Warning: add_info_to_partition: add info to {n_tot} images')   
    return partition

def get_partition(config_file):
    conf = read_from_yaml(config_file)
    conf_p = conf.partition
    if conf_p.partition_file and os.path.exists(conf_p.partition_file) and not conf_p.overwrite:
        print(f'Warning: get_partition: read partition from {conf_p.partition_file}')
        return read_json(conf_p.partition_file)

    if conf_p.old_parition_order:# use an old fixed partition
        partition = get_old_partition(conf_p.old_parition_order)
    elif conf.images.root:# make a new partition
        # root_data, data_paths = get_data_paths(config_file, data_type='images')
        root_data, data_paths = get_data_paths(config_file, data_type='images')
        ids, fractions = list(data_paths.keys()), conf_p
        n_data = len(ids)
        n_train = np.uint(np.ceil(n_data * fractions.training))
        n_validate = np.uint(np.floor(n_data * fractions.validation))
        n_test = np.uint(n_data - n_train - n_validate)
        assert n_test > 0, 'Not enough data left for testing'
        # Random shuffle
        ids = sample(list(ids), n_data)
        n_split = n_train + n_validate
        splits = {'training': (0, n_train),
                  'validation': (n_train, n_split),
                  'testing': (n_split, n_data)}
        partition = {subset: ids[span[0]:span[1]] for subset, span in splits.items()}
    else:
        raise NotImplementedError

#     if conf.images.root:
    root_data, data_paths = get_data_paths(config_file, data_type='images')
    root_tile, tile_paths = get_data_paths(config_file, data_type='tiles')
    
    partition = add_paths_to_partition(partition, data_paths=data_paths, tile_paths=tile_paths)
#     if conf.tiles.root:  
#         partition = add_paths_to_partition(partition, tile_paths=tile_paths)
    partition = add_info_to_partition(root_data, partition,type_='label')
    for type_ in namespace_to_dict(conf.tiles.subdirs).keys():
        if type_ == 'image':
            continue
        partition = add_tile_ratio_to_partition(root_tile, partition, type_=type_)
    

    try_create_dir(os.path.split(conf_p.partition_file)[0])
    save_json(conf_p.partition_file, partition)
    print(f'Warning: get_partition: save partition_file to {conf_p.partition_file}')
    return partition


def get_labels(config_file):
    conf = read_from_yaml(config_file).label
    if conf.label_file and os.path.exists(conf.label_file) and not conf.overwrite:
        print(f'Warning: get_labels: read labels from {conf.label_file}')
        return read_json(conf.label_file)
    labels = get_data_labels(config_file, conf.binary)
    try_create_dir(os.path.split(conf.label_file)[0])
    save_json(conf.label_file, labels)
    print(f'Warning: get_labels: save label_file to {conf.label_file}')
    return labels




def main():
    config_file = '../../configs/data_to_partition.yml'
    # data_paths = get_data_paths(config_file)
    get_partition(config_file)
    l = get_labels(config_file)
    # print(l)
    # get_tile_ratios(partition)
    # partition = add_tile_ratios_to_partition(partition)
    pass


if __name__ == '__main__':
    main()
