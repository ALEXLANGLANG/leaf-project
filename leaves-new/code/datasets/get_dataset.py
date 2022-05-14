from torch.utils.data import ConcatDataset
from torchvision import transforms

from tool.data_io import read_json
from datasets.leaf_dataset import BinaryLeaf

from transforms.transform import get_transforms, ToTensor
from tool.yaml_io import read_from_yaml


def get_dataset(cfg):
    cfg_d = cfg.data
    
    cfg_trans = cfg.training.transform
    list_train_datasets = []
    list_valid_datasets = []
    evaluate_phase = cfg.evaluate.phase
    first_val=True
    for root, partition_file in zip(cfg_d.root, cfg_d.partition_file):
        partition = read_json(partition_file)
        if 'training' in partition.keys():
            trans_train = transforms.Compose(get_transforms(cfg_trans))
            train_data = BinaryLeaf(root, 'training', partition_file, cfg_d.label_file,
                                    transform=trans_train, 
                                    label_ratio=cfg_d.label_ratio, 
                                    leaf_mask_ratio=cfg_d.leaf_mask_ratio,
                                    data_type = cfg_d.data_type
                                   )
            list_train_datasets += [train_data]
        
        if evaluate_phase in partition.keys():
            if first_val:
                trans_valid = transforms.Compose(get_transforms(cfg_trans, False))
                valid_data = BinaryLeaf(root, evaluate_phase, partition_file, cfg_d.label_file,
                                        transform=trans_valid, label_ratio=-1, leaf_mask_ratio=-1,
                                        data_type = cfg_d.data_type)
                list_valid_datasets = [valid_data] # only use the first valid dataset
                first_val=False
                
    if len(list_train_datasets)!=0:
        train_data = list_train_datasets[0] if len(list_train_datasets) == 1 else ConcatDataset(list_train_datasets)
    else:
        train_data = None
    if len(list_train_datasets)!=0:
        valid_data = list_valid_datasets[0] if len(list_valid_datasets) == 1 else ConcatDataset(list_valid_datasets)
    else:
        valid_data = None
    return train_data, valid_data


def main():
    cfg = read_from_yaml('../../configs/config.yml')
    train_data, valid_data = get_dataset(cfg)
    print(len(train_data))
    print(len(valid_data))
    pass


if __name__ == '__main__':
    main()
