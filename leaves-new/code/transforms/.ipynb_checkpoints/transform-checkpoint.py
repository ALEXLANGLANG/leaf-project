import random

from PIL import Image
from torchvision import transforms
import numpy
import torch
import copy
import numpy as np
import sys
import cv2
from path import Path
sys.path.append('../')


from tool.data_io import read_image, save_image
from tool.yaml_io import namespace_to_dict

def rotate_image(image, angle, cval=0):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    if type(cval) == int:
        if image.ndim == 3:
            cval = 3 * [cval]
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST, borderValue=cval)
    return result


def process_dict_of_imgs(dict_imgs, func, **kwargs):
    res = copy.deepcopy(dict_imgs)
    for key in res.keys():
        if isinstance(res[key], numpy.ndarray) or isinstance(res[key], torch.Tensor):
#             print("before",res[key].shape)
            res[key] = func(res[key], **kwargs)
#             print("after",res[key].shape)
    return res


class RandomFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    the class of the input should be numpy array or a dictionary that contain numpy arrays
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        axis:
            None or int.
    """

    def __init__(self, p_h=0.5, p_v=0.5, axis=None):
        super().__init__()
        self.p_h = p_h
        self.p_v = p_v
        self.axis = axis

    def forward(self, imgs):
        """
        Args:
            imgs (Numpy array or dictionary of numpy array): Image to be flipped.

        Returns:
            Numpy array or dictionary of numpy arrays: Randomly flipped images.
        """

        if self.axis is not None:
            return flip_image(imgs, axis=self.axis) if type(imgs) == numpy.ndarray else process_dict_of_imgs(imgs,
                                                                                                             flip_image,
                                                                                                             axis=self.axis)
        self.axis = None if random.random() < self.p_h else 0
        axis_extra = None if random.random() < self.p_v else 1
        if type(imgs) == numpy.ndarray:
            return flip_image(imgs, axis=self.axis, axis_extra=axis_extra)
        elif type(imgs) == dict:
            return process_dict_of_imgs(imgs, flip_image, axis=self.axis, axis_extra=axis_extra)
        else:
            raise ValueError('input should be a numpy array or a dictionary of numpy arrays')
        return imgs


def flip_image(img, axis=None, axis_extra=None):
    if axis:
        img = np.flip(img, axis=axis)
    if axis_extra:
        img = np.flip(img, axis=axis_extra)
    return img


class RandomColorJitter(torch.nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.bright = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, imgs):
        """
        Args:
            imgs (Tensor)
        Returns:
            Tensor after color jtter
        """
        colorJitter = transforms.ColorJitter(brightness=self.bright, contrast=self.contrast, saturation=self.saturation,
                                             hue=self.hue)
        if type(imgs) == torch.Tensor:
            imgs = colorJitter(imgs)
        elif type(imgs) == dict:
            imgs['image'] = colorJitter(imgs['image'])
        else:
            raise ValueError('input should be a numpy array or a dictionary of numpy arrays')
        return imgs


class RandomRotation90(torch.nn.Module):
    def __init__(self, axes=(0, 1), k=None):
        super().__init__()
        self.axes = axes
        self.k = k

    def forward(self, imgs):
        """
        Args:
            imgs (Numpy array or dictionary of numpy array): Image to be flipped.

        Returns:
            Numpy array or dictionary of numpy arrays: Randomly flipped images.
        """
        if self.k is None:
            self.k = random.randint(1, 4)
        if type(imgs) == numpy.ndarray:
            imgs = np.rot90(imgs, k=self.k, axes=self.axes, dtype=imgs.dtype)
        elif type(imgs) == dict:
            imgs = process_dict_of_imgs(imgs, np.rot90, k=self.k, axes=self.axes)
        else:
            raise ValueError('input should be a numpy array or a dictionary of numpy arrays')
        return imgs


def convert_Tensor_to_numpy_img(img):
    img = torch.squeeze(img)
    if img.ndim == 2:
        img = torch.unsqueeze(img, dim=0)
    img = img.detach().permute(1, 2, 0).cpu().numpy()
    return img


def convert_img_numpy_to_Tensor(img):
    if type(img) == numpy.ndarray:
        if img.dtype == bool:
            img = torch.from_numpy(img).type(torch.bool).permute(2, 0, 1)
        else:
            img = torch.from_numpy(img).type(torch.float32).permute(2, 0, 1)
    elif isinstance(img, torch.Tensor):
        return img
    else:
        print(f'Error: convert_img_numpy_to_Tensor: Invalid type')
        exit(1)
    return img


class ToTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        
        if type(sample) == numpy.ndarray:
            return convert_img_numpy_to_Tensor(sample)
        elif type(sample) == dict:
            return process_dict_of_imgs(sample, convert_img_numpy_to_Tensor)
        else:
            raise ValueError('input should be a numpy array or a dictionary of numpy arrays')
        return process_dict_of_imgs(sample, convert_img_numpy_to_Tensor)

    def _get_name(self):
        return 'ToTensor'

    
class Resize(torch.nn.Module):
    def __init__(self, m = 1024, n = 1024, interpolation = cv2.INTER_NEAREST, scale = None):
        super().__init__()
        self.m = m
        self.n = n
        self.interpolation = interpolation
        self.scale = scale
    def forward(self, imgs):
        """
        Args:
            imgs (numpy)
        Returns:
            resized imgs
        """
        if type(imgs) == numpy.ndarray:
            m, n = self.m, self.n
            if imgs.dtype == bool:
                imgs = imgs.astype(float)
            if self.scale is not None:
                img_m, img_n = imgs.shape[0], imgs.shape[1]
                assert self.scale > 0 and self.scale <= 1, '0 < scale <= 1'
                m, n = int(self.scale*img_m), int(self.scale*img_n)
                m = m + 1 if m % 2 == 1 else m
                n = n + 1 if n % 2 == 1 else n
            imgs = cv2.resize(imgs, dsize=(n, m), interpolation=self.interpolation)
            if imgs.ndim == 2:
                imgs = np.expand_dims(imgs, axis=2)
            return imgs
        elif type(imgs) == dict:
            for key in imgs.keys():
                if isinstance(imgs[key], numpy.ndarray) or isinstance(imgs[key], torch.Tensor):
                    imgs[key] = self.forward(imgs[key])
        else:
            raise ValueError('input should be a numpy array or a dictionary of numpy arrays')
        return imgs
    

def print_transform_info(list_trans, header=''):
    msg = f'{header} Transforms are '
    for trans in list_trans:
        msg += f'{trans._get_name()}, '
    print(msg)


def get_transforms(aug, train = True):
    list_ = []
    name2bool = namespace_to_dict(aug)
    if 'resize' in name2bool.keys() and aug.resize is not None:
        list_.append(Resize(aug.resize.m, aug.resize.n, scale =  aug.resize.scale))
    if 'rotation90' in name2bool.keys() and aug.rotation90 and train:
        list_.append(RandomRotation90())
    if 'flip' in name2bool.keys() and aug.flip and train:
        list_.append(RandomFlip())
    if 'toTensor' in name2bool.keys() and aug.toTensor:
        list_.append(ToTensor())
    if 'color_jitter' in name2bool.keys() and aug.color_jitter and train:
        list_.append(RandomColorJitter(brightness=0.2, contrast=0.2, hue=0.08, saturation=0.2))
    return list_


def test_rotate():
    def read_image_with_palette(path, resize_shape=None):
        try:
            with Image.open(path) as img:
                palette = img.getpalette()
                if resize_shape is not None:
                    img = img.resize(resize_shape)
                idx = np.array(img)
                return idx, palette
        except:
            return None, None
    root = Path('../../../data_new/leaves/quercus-lobata')
    img_path = Path('images_clean/00000335_2038640.jpeg')
    label_path = Path('releases/1.0/other_formats/semantic-mask/masks_clean/00000335_2038640.png')
    new_root = Path('../../../../meeting/10_29/imgs')
    img, _ = read_image(root/img_path)
    save_image(new_root/'img_00000335_2038640.jpeg',img)
    label, p = read_image(root/label_path)
    save_image(new_root / 'label_00000335_2038640.png', label, p)
    img_r = rotate_image(img, 45, 255)
    save_image(new_root / 'imgR_00000335_2038640.jpeg', img_r)
    label_r = rotate_image(label, 45, 0)
    save_image(new_root / 'labelR_00000335_2038640.png', label_r, p)
    import IPython;
    IPython.embed()
    exit()
    # plt_samples([img, img_r, label, label_r], n_r=2, n_c=2)


    path='../../../../meeting'
def main():
    test_rotate()
    # path = '../../../new_configs/config.yml'
    # aug = read_from_yaml(path).experiment.training.transform
    # print(get_transforms(aug))


if __name__ == '__main__':
    main()

