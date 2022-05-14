import glob
import os


import numpy as np
from tqdm import tqdm


import sys
sys.path.append('../')
from tool.yaml_io import read_from_yaml
from tool.data_io import get_data_paths, read_image, save_image, try_create_dir


def number_of_tiles(image_size, tile_size, tile_margin):
    assert image_size >= tile_size, 'the tile is bigger than the image'
    tile_overlap = 2 * tile_margin
    tile_pitch = tile_size - tile_overlap
    assert tile_pitch > 0, 'too much overlap for this tile size'
    n_tiles = np.ceil((image_size - tile_overlap) / tile_pitch).astype(int)
    return n_tiles, tile_pitch



def save_tiles(path, base_name, dir_tile, type_, t_cols, t_rows, t_margin,start_point=(0,0),  overwrite=False):
    img, plate = read_image(path)
    i_rows, i_cols = img.shape[0], img.shape[1]
    n_tiles = 1
    start_r, start_c = start_point
    assert i_cols>=t_cols and i_rows>=t_rows
    start_c = int(i_cols - t_cols) if int(i_cols-start_c) < t_cols else start_c
    start_r = int(i_rows - t_rows) if int(i_rows-start_r) < t_rows else start_r
    n_c, c_pitch = number_of_tiles(int(i_cols-start_c), t_cols, t_margin)
    n_r, r_pitch = number_of_tiles(int(i_rows-start_r), t_rows, t_margin)
    r_start = (i_rows - t_rows) // 2 if n_r == 1 else start_r
    for row in range(n_r):
        if n_r > 1 and row == n_r - 1:
            r_start = i_rows - t_rows
        r_stop = r_start + t_rows
        c_start = (i_cols - t_cols) // 2 if n_c == 1 else start_c
        for column in range(n_c):
            if n_c > 1 and column == n_c - 1:
                c_start = i_cols - t_cols
            c_stop = c_start + t_cols
            path_tile = os.path.join(dir_tile, f'{base_name}_{n_tiles}-{r_start}-{r_stop}-{c_start}-{c_stop}' + type_)
            n_tiles += 1
            if not os.path.exists(path_tile) or overwrite:
#                 res += [img[r_start:r_stop, c_start:c_stop]]
                save_image(path_tile, img[r_start:r_stop, c_start:c_stop], plate)
            c_start += c_pitch
        r_start += r_pitch
    return None

def gen_save_data_tiles(config_file):
    root_img, data_paths = get_data_paths(config_file, data_type='images')
    conf = read_from_yaml(config_file).tiles
    for img_id in tqdm(data_paths):
        paths = data_paths[img_id]
        for (file_type, relpath) in paths.items():
            species, base_name = os.path.split(img_id)
            dir_tile = os.path.join(conf.root, species, file_type, base_name)
            path = os.path.join(root_img, relpath)
            try_create_dir(dir_tile)
            type_ ='.jpeg' if file_type=='image' else '.png'
            save_tiles(path, base_name, dir_tile, type_, conf.width, conf.height, conf.margin,start_point=conf.start_point, overwrite=conf.overwrite)


def main():
    # import IPython; IPython.embed()
    # exit()
    config_file = '../../configs/data_to_tiles.yml'
    gen_save_data_tiles(config_file)


if __name__ == '__main__':
    main()

 # def save_tiles(path, base_name, dir_tile, type_, t_cols, t_rows, t_margin, overwrite=False):
#     img, plate = read_image(path)
#     i_rows, i_cols = img.shape[0], img.shape[1]
#     n_tiles = 1
#     n_c, c_pitch = number_of_tiles(i_cols, t_cols, t_margin)
#     n_r, r_pitch = number_of_tiles(i_rows, t_rows, t_margin)
#     r_start = (i_rows - t_rows) // 2 if n_r == 1 else 0
#     for row in range(n_r):
#         if n_r > 1 and row == n_r - 1:
#             r_start = i_rows - t_rows
#         r_stop = r_start + t_rows
#         c_start = (i_cols - t_cols) // 2 if n_c == 1 else 0
#         for column in range(n_c):
#             if n_c > 1 and column == n_c - 1:
#                 c_start = i_cols - t_cols
#             c_stop = c_start + t_cols
#             path_tile = os.path.join(dir_tile, f'{base_name}_{n_tiles}-{r_start}-{r_stop}-{c_start}-{c_stop}' + type_)
#             n_tiles += 1
#             if not os.path.exists(path_tile) or overwrite:
#                 save_image(path_tile, img[r_start:r_stop, c_start:c_stop], plate)
#             c_start += c_pitch
#         r_start += r_pitch
