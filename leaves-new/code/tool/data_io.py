import csv
import glob
import json
import os
import sys

from path import Path
sys.path.append('../')

from PIL import Image
from tool.colors import get_raw_arrays
from tool.yaml_io import read_from_yaml, namespace_to_dict

import tool.colors as lut  # For "lookup table"


def get_data_labels(config_file, is_binary=True):
    def labels_from_file(file_name):
        ids = {}
        with open(file_name, newline='') as file:
            reader = csv.DictReader(file, delimiter=',')
            for number, row in enumerate(reader):
                ids[row['class_name']] = {'number': number, 'color': row['class_color']}
        return ids

    cnf = read_from_yaml(config_file).images
    data_labels = {species: labels_from_file(Path(cnf.root) / species / cnf.subdir_labelmap) for species in cnf.dirs}
    data_labels = common_data_labels(data_labels, is_binary)
    return data_labels


# Flip the label map (from number to label name and a new number,
# which is what is needed for remapping), and redefine label numbers and colors
# to be consistent across species.
# Only retain labels common to all species. The others become background labels.
def common_data_labels(maps, is_binary=True):
    common_labels = set.intersection(*[set(sp_map.keys()) for (sp, sp_map) in maps.items()])
    black = [0, 0, 0]
    new_values = {'__background__': {'number': 0, 'color': list(black)}}
    non_background_labels = common_labels.copy()
    non_background_labels.remove('__background__')
    if is_binary:
        color = [255] * 3
        for number, label in enumerate(non_background_labels):
            new_values[label] = {'number': 1, 'color': list(color)}
    else:
        colors = lut.distinctive_colors(len(non_background_labels))
        for number, label in enumerate(non_background_labels):
            color = colors[number]
            new_values[label] = {'number': number + 1, 'color': [int(c) for c in color]}
    new_maps = {}
    for species in list(maps.keys()):
        new_map = {}
        for label, value in maps[species].items():
            if label in common_labels:
                new_map[value['number']] = {'label': label,
                                            'number': new_values[label]['number'],
                                            'color': new_values[label]['color']}
            else:
                new_map[value['number']] = {'label': '__background__',
                                            'number': new_values['__background__']['number'],
                                            'color': new_values['__background__']['color']}
        new_maps[species] = new_map
    print(f'Warning: common_data_labels(is_binary_mode={is_binary}): use common labels {common_labels}')
    return new_maps


def get_data_paths(config_file='./data_to_tiles.yml', data_type='images'):
    conf = read_from_yaml(config_file).images if data_type == 'images' else read_from_yaml(config_file).tiles
    data_paths = {}
    for species in conf.dirs:
        for file_type, dir_ in namespace_to_dict(conf.subdirs).items():
            for path in sorted(glob.glob(os.path.join(conf.root, species, dir_))):
                if data_type == 'images':
                    base_name = os.path.basename(path).split('.')[0]
                    data_id = os.path.join(species, base_name)
                elif data_type == 'tiles':
                    data_id = os.path.join(species, os.path.split(path)[1].split('.')[0])
                    
                if data_id not in data_paths.keys():
                    data_paths[data_id] = {}
                if len(data_paths[data_id]) == 0:
                    data_paths[data_id] = {file_type: os.path.relpath(path, conf.root)}
                else:
                    data_paths[data_id][file_type] = os.path.relpath(path, conf.root)
                    
    data_paths = checking_data_paths(data_paths, len(namespace_to_dict(conf.subdirs)), path=conf.root, name=data_type)
    return conf.root, data_paths


# Check if each sample has the same number of types like image, label, leaf_mask
def checking_data_paths(data_paths, n_paths, path, name='images'):
    dict_com = {}
    first_id = None
    for _id in data_paths:
        first_id = _id if first_id is None else first_id
        paths = data_paths[_id]
        if n_paths == len(paths):
            dict_com[_id] = paths
    n_img = len(data_paths)
    print(f'Warning: get_data_paths: read {len(dict_com)} {name} with {list(data_paths[first_id].keys())} from {path}')
    if n_img != len(dict_com):
        print(f'Warning: checking_data_paths: {n_img} {name} in raw data '
              f'but use {len(dict_com)} common {name} with {list(data_paths[first_id].keys())}')
    return dict_com


# read one image from the path and convert it to numpy type
def read_image(path):
    try:
        with Image.open(path) as img:
            return get_raw_arrays(img)
    except:
        print("**** read_image: cannot find {}".format(path))
    return None, None


def save_image(path, img, palette=None):
    im = Image.fromarray(img)
    if palette is not None:
        im.putpalette(palette)
    im.save(path)


# create a directory if this dir does not exist
def try_create_dir(directory, verbose=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except:
            if verbose:
                print('create_dir: cannot makedirs for {}'.format(directory))
    else:
        if verbose:
            print("Dir exists: {}".format(directory))


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def test():
    path = "../../../data_new/leaves/onoclea-sensibilis-herbivory/images_clean/00001194_5679034.jpeg"
    root = "../../../data_new/leaves/"
    cnf = '../../../new_configs/data_to_tiles.yml'
    file_name = '../../../data_new/leaves/quercus-lobata/releases/1.0/other_formats/semantic-mask/class_mapping.csv'
    maps = get_data_labels(cnf)


if __name__ == '__main__':
    test()
