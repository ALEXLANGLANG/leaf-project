import glob
import os


import numpy as np
from tqdm import tqdm
import cv2

import sys
sys.path.append('../')
from tool.yaml_io import read_from_yaml,namespace_to_dict
from tool.data_io import get_data_paths, read_image, save_image,read_json, try_create_dir

def rotate_image(image, angle,borderMode=cv2.BORDER_REFLECT, borderValue=0):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], 
                            flags=cv2.INTER_NEAREST,
                           borderMode =borderMode,  borderValue=borderValue)
    return result
def resize_image(image, ):
    
def save_rotate_imgs(config_file):
    root_img, data_paths = get_data_paths(config_file, data_type='images')
    conf = read_from_yaml(config_file).rotate
    for img_id in tqdm(data_paths):
        paths = data_paths[img_id]
        for (file_type, relpath) in paths.items():
            species, base_name = os.path.split(img_id)
            path = os.path.join(root_img, relpath)
            old_dir, old_name = os.path.split(path)
            new_dir = old_dir + conf.suffix+str(conf.angle)
            try_create_dir(new_dir)
            path_saved = os.path.join(new_dir, old_name)
            if not os.path.exists(path_saved) or conf.overwrite:
                img, plate = read_image(path)
                bool_type = True if img.dtype == bool else False
                if bool_type:
                    img = img.astype('float')
                img_rot=rotate_image(img, angle=conf.angle,borderMode=cv2.BORDER_REFLECT)
                if bool_type:
                    img = img.astype('bool')
                    img_rot=img_rot.astype('bool')

                save_image(path_saved,img_rot,plate)
#             
def main():
    config_file = '../../configs/data_to_rotate.yml'
    save_rotate_imgs(config_file)
    pass

if __name__ =='__main__':
    main()
