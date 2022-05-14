from matplotlib import pyplot as plt

from tool.yaml_io import namespace_to_dict


def plt_samples(imgs, n_r, n_c, figsize=(10, 10), title='', cmap='gray',axis=False):
    if type(imgs) == dict:
        imgs = list(imgs.values())
    ct = 1
    plt.figure(figsize=figsize)
    plt.suptitle(title)
    for c in range(n_c):
        for r in range(n_r):
            plt.subplot(n_r, n_c, ct)
            if not axis:
                plt.axis('off')
            plt.imshow(imgs[int(ct - 1)], cmap=cmap)
            ct += 1

    plt.show()

def print_dict(dic, msg=''):
    if type(dic)!=dict:
        dic = namespace_to_dict(dic)
    for k, v in dic.items():
        if type(v) == dict:
            msg += f'\n--{k}:  '
            msg = print_dict(v, msg)
        else:
            msg += f'{k}:{v} | '
    return msg
