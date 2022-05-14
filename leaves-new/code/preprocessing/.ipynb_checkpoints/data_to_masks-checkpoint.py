import os

import numpy as np
from skimage import morphology
from tqdm import tqdm
import sys
sys.path.append('../')

from tool.data_io import get_data_paths, read_image, save_image, try_create_dir
from path import Path
# Helper function to get_edge_leaf
from tool.yaml_io import read_from_yaml


# This function is used in two branches method
# It outputs the mask of interior leaf so that the model will only
# work on the interior of leaf.
def get_interior_leaf(edge_leaf_mask, whole_leaf_mask):
    interior_leaf_mask = np.logical_and(~edge_leaf_mask, whole_leaf_mask)
    return interior_leaf_mask


def get_edge_leaf_helper(img_leaf_mask, width=20, fill_val=False):
    """

    Args:
        img_leaf_mask: This should be binary image mask:  leaf:label1, background: label0
        width: the width of edge

    Returns:
        A binary mask: edge_leaf:label1, rest:label0

    """
    m, n = img_leaf_mask.shape
    diff_img_leaf_mask_axis0 = np.diff(img_leaf_mask, axis=0)
    indices = np.nonzero(diff_img_leaf_mask_axis0)
    leaf_mask_new = img_leaf_mask.copy()
    for r, c in zip(indices[0], indices[1]):
        c_stop = c + width if c + width else n - 1
        c_start = c - width if c - width else 0
        r_stop = r + width if r + width else m - 1
        r_start = r - width if r - width else 0
        leaf_mask_new[r_start:r_stop, c_start:c_stop] = fill_val
    bw = np.logical_xor(leaf_mask_new, img_leaf_mask)
    return bw


# the edge as the naive model think everything on the edge as damage
def get_edge_leaf_mask(path_input, width_leaf=20, width_bg=5):
    """

    Args:
        width_bg: width of white background
        width_leaf: width of leaf
        img_leaf_mask: This should be binary image mask:  leaf:label1, background: label0
    Returns:
        A binary mask: edge_leaf:label1, rest:label0

    """
    img_rgb, _ = read_image(path_input)
    # img_leaf_mask = get_leaf_mask_helper(img_rgb, 250)
    non_leaf_mask = np.mean(img_rgb, axis=2) > 250
    img_leaf_mask = (non_leaf_mask == False)
    bw_0 = get_edge_leaf_helper(img_leaf_mask, width_leaf, fill_val=False)
    bw_1 = get_edge_leaf_helper(img_leaf_mask, width_bg, fill_val=True)
    return np.logical_or(bw_0, bw_1)


def get_leaf_mask(path_input, threshold=250, threshold_closed_object_size=1e6):
    img_rgb,_ = read_image(path_input)
    non_leaf_mask = np.mean(img_rgb, axis=2) > threshold
    leaf_mask = (non_leaf_mask == False)
    leaf_mask_original = leaf_mask.copy()
    num_pos = np.count_nonzero(leaf_mask)
    object_size = min(num_pos / 20, threshold_closed_object_size)
    leaf_mask = morphology.remove_small_objects(leaf_mask, object_size)
    leaf_mask = morphology.remove_small_holes(leaf_mask, object_size)
    leaf_mask = np.logical_or(leaf_mask,
                              leaf_mask_original)  # Make sure remove_small_holes will not hurt any leaves
    edge_leaf = get_edge_leaf_mask(path_input, width_leaf=15,
                                   width_bg=5)  # Give the mask some white background edge
    leaf_mask = np.logical_or(leaf_mask, edge_leaf)
    return leaf_mask


def gen_save_data_masks(config_file):
    root, data_paths = get_data_paths(config_file=config_file)
    root = Path(root)
    conf = read_from_yaml(config_file).masks
    print(f'Warning: gen_save_data_masks: convert data to {conf.type}')
    for img_id in tqdm(data_paths.keys()):
        paths = data_paths[img_id]
        dir_, base_name = os.path.split(paths['image'])

        path_mask = os.path.join(root, dir_ + conf.suffix, base_name.split('.')[0] + '.png')

        if conf.type == 'leaf_mask':
            mask = get_leaf_mask(root/paths['image'], conf.leaf_mask.threshold_ostu,
                                 conf.leaf_mask.threshold_closed_object_size)
        elif conf.type == 'edge_mask':
            mask = get_edge_leaf_mask(root/paths['image'], conf.edge_mask.width_leaf, conf.edge_mask.width_bg)
        else:
            print(f'please select type from (edge_mask leaf_mask)')
            exit(1)
        if os.path.exists(path_mask) and not conf.overwrite:
            continue
        try_create_dir(os.path.split(path_mask)[0])
        save_image(path_mask, mask)

def main():
    config_file = '../../configs/data_to_masks.yml'
    gen_save_data_masks(config_file)



if __name__ == '__main__':
    main()
