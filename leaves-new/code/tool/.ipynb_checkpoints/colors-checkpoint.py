# Visually distinctive colors fron Sasha Trubetskoys post
# https://sashamaps.net/docs/tools/20-colors/ .
# They are arranged so that picking the first n yields the n most
# distinctive ones

from PIL import Image
import numpy as np


trubetskoy = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
              (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
              (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
              (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
              (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
              (255, 255, 255), (0, 0, 0)]


# Returns an n by 3 numpy array of np.uint8 RGB values, for n <= 22.
# White and black are colors 20 and 21
def distinctive_colors(n=None):
    max_n = len(trubetskoy)
    if n is None:
        n = max_n
    msg = 'Can only provide up to {} distinct colors, but {} were requested'
    assert n <= max_n, msg.format(max_n, n)
    return np.array(trubetskoy[:n], dtype=np.uint8)


def get_raw_arrays(image):
    """Extract index array and color map"""
    idx = np.array(image)
    palette = image.getpalette()
    if palette is None:
        lut = None
    else:
        lut = np.array(palette, dtype='uint8')
        lut = np.reshape(lut, (len(lut) // 3, 3))
    return idx, lut



def recolor(idx, lut):
    image = Image.fromarray(idx, mode='P')
    image.putpalette(np.ravel(lut), rawmode='RGB')
    return image


def get_rgb_array(image):
    return np.array(image.convert('RGB'))


def convert_rgb_to_gray(img):
    return img.mean(axis=2)



