import glob
import os.path

from tqdm import tqdm
import sys
sys.path.append('../')

from leavesImageEditor.adjustFrame import AdjustFrame
from tool.data_io import get_data_paths, read_json, save_json, try_create_dir, save_image, read_image
from tool.yaml_io import read_from_yaml


def read_clean_info(config_file):
    """
    Args:
        config_file:
    Returns:
        A dictoionary: [img_id, actions]
        actions means all the processing related to this image, including
        cropping, removing, threshold adjusting
    """

    clean = read_from_yaml(config_file).clean
    if os.path.exists(clean.clean_info_file) and not clean.overwrite:
        print(f'read clean_info from {clean.clean_info_file}')
        return read_json(clean.clean_info_file)

    clean_infos = {}
    for path in glob.glob(clean.paths):
        clean_info = read_json(path)
        clean_infos[clean_info['img_id']] = clean_info['actions']
    try_create_dir(os.path.split(clean.clean_info_file)[0])
    save_json(clean.clean_info_file, clean_infos)
    print(f'Warning: read_clean_info: save clean_info to {clean.clean_info_file}')
    return clean_infos


def save_clean_img(path_img, path_label, suffix, acts,overwrite):
    def read_coords(coords_info):
        return coords_info['r_start'], coords_info['r_stop'], coords_info['c_start'], coords_info['c_stop']

    def get_new_path(path, suffix_, type_=None):
        dir_, base_name = os.path.split(path)
        type_ = base_name.split('.')[1] if type_ is None else 'jpeg'
        return os.path.join(dir_ + suffix_, base_name.split('.')[0] +'.'+ type_)

    path_clean_img = get_new_path(path_img, suffix, 'jpeg')
    path_clean_label = get_new_path(path_label, suffix)
    if os.path.exists(path_clean_img) and os.path.exists(path_clean_label) and not overwrite:
        return

    img, _ = read_image(path_img)
    label, palette = read_image(path_label)
#     from tool.plt_utils import plt_samples
    for act_id in range(len(acts)):
        act_id = str(act_id)
        if 'crop' in acts[act_id].keys():
            if acts[act_id]['crop']['type'] == 'keep':
                r1, r2, c1, c2 = read_coords(acts[act_id]['crop'])
                img = img[r1:r2, c1:c2]
                label = label[r1:r2, c1:c2]
            elif acts[act_id]['crop']['type'] == 'remove':
                r1, r2, c1, c2 = read_coords(acts[act_id]['crop'])
                img[r1:r2, c1:c2] = [255, 255, 255]
        elif 'adjust' in acts[act_id].keys():
            thres = acts[act_id]['adjust']['threshold']
            object_size = acts[act_id]['adjust']['object size']
            outline_width = acts[act_id]['adjust']['outline_width'] if 'outline_width' in acts[act_id][
                'adjust'].keys() else 1
            img, _, _, _ = AdjustFrame.preprocess_a_image(img, thres, object_size, outline_width)
            label = AdjustFrame.recolor_outline(label, outline_width, 0)

    try_create_dir(os.path.split(path_clean_img)[0])
    try_create_dir(os.path.split(path_clean_label)[0])
    save_image(path_clean_img, img)
    save_image(path_clean_label, label, palette)
    


def gen_save_data_clean(config_file):
    clean_infos = read_clean_info(config_file)
    root, data_paths = get_data_paths(config_file)
    clean = read_from_yaml(config_file).clean
    deleted_id = read_from_yaml(config_file).deleted_id
    for img_id in tqdm(data_paths):
        paths = data_paths[img_id]
        if img_id in clean_infos.keys():
            if img_id in deleted_id:
                print(f"Ignore {img_id} as it is in deleted_ids")
            else:
                save_clean_img(os.path.join(root, paths['image']), os.path.join(root, paths['label']), clean.suffix, clean_infos[img_id],clean.overwrite)
        else:
            print("Warning: not found: ", img_id)


def main():
    config_file = '../../configs/data_to_clean.yml'
    gen_save_data_clean(config_file)

    pass


if __name__ == '__main__':
    main()
