from losses.filter_loss import FilterLossWithLogits
from losses.focal_loss import BinaryFocalLossWithLogits
from tool.yaml_io import read_from_yaml


def get_loss(cfg):
    l, d = cfg.training.loss, cfg.data
    if l.name == 'focal':
        loss = BinaryFocalLossWithLogits(alpha=l.alpha, gamma=l.gamma,
                                         reduction='mean')
    elif l.name == 'filter':
        loss = FilterLossWithLogits(a0=1-l.alpha, a1=l.alpha, b1=l.b1, beta=l.beta, hinge='ReLU', margin = l.margin, reduction='mean')
    else:
        raise NotImplementedError(l.name)
    return loss


def main():
    cfg = read_from_yaml('../../../new_configs/config.yml')
    print(get_loss(cfg))
    pass


if __name__ == '__main__':
    main()
