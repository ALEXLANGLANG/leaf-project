from tool.yaml_io import read_from_yaml
from trainer import leaf_trainer


def get_trainer(cfg):
    t = cfg.training.trainer
    if t.name == 'binaryLeaf':
        trainer =leaf_trainer.TrainFramework
    else:
        raise NotImplementedError(t.name)
    return trainer


def main():
    cfg = read_from_yaml('../../../new_configs/config.yml')
    print(get_trainer(cfg))
    pass


if __name__ == '__main__':
    main()
