from models.TernausNet.UnetVGG import UNet11, UNet16
from models.unet.unet_model import UNet
from tool.yaml_io import read_from_yaml


def get_model(m):
    if m.name == 'Unet':
        net = UNet(n_channels=m.n_channels, n_classes=m.n_classes, bilinear=m.bilinear)
    elif m.name == 'Pretrained_Unet11':
        net = UNet11(n_classes=m.n_classes, pretrained=True)
    elif m.name == 'Pretrained_Unet16':
        net = UNet16(n_classes=m.n_classes, pretrained=True)
    else:
        raise NotImplementedError(m.name)
    return net


def main():
    cfg = read_from_yaml('../../configs/config.yml')
    print(get_model(cfg))
    pass
if __name__=='__main__':
    main()
