import random
import sys
sys.path.append('../')
import numpy as np
from path import Path
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from tool.data_io import read_image, read_json
from transforms.transform import get_transforms, convert_Tensor_to_numpy_img
from tool.plt_utils import  plt_samples
from tool.yaml_io import read_from_yaml, namespace_to_dict


class ImgSeqDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, phase, partition_file, label_file, transform=None, ignore_index=None):
        self.msg = ''
        self.ignore_index = ignore_index
        self.root = Path(root)
        self.phase = phase
        self.partition_file = partition_file
        self.transform = transform
        self.label_file = label_file
        self.samples = self.collect_samples()
        self.labels = self.set_labels()

    @abstractmethod
    def collect_samples(self):
        pass

    @abstractmethod
    def set_labels(self):
        pass

#     @abstractmethod
#     def set_ignore_index(self, batch):
#         pass

    def _load_sample(self, idx):
        batch_info = self.samples[idx]
        species = batch_info['id'].split('/')[0]  
        batch = {
            'idx': int(idx),
            'id': batch_info['id']
        }
        for type_ in batch_info.keys():
            if 'info' in type_ or 'id' in type_ or 'ratio' in type_:
                continue 
            sample = read_image(self.root / batch_info[type_])[0].astype(np.float32)
            if type_ == 'image':
                sample = sample.astype(float) / 255.
            elif type_ == 'label':
                for key, value in self.labels[species].items():
                    sample[sample == int(key)] = value['number']
            elif 'mask' in type_:
                sample = sample.astype(bool)
            if sample.ndim == 2:
                sample = np.expand_dims(sample, axis=2)
            batch[type_] = sample
        return batch

    def get_batch(self, index):
        batch = self.__getitem__(index)
        for key in batch.keys():
            if type(batch[key]) == torch.Tensor:
                batch[key] = torch.unsqueeze(batch[key], dim=0)
        return batch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        batch = self._load_sample(idx)
        if self.transform:
#             print(batch.keys())
            batch = self.transform(batch)
        return batch


class BinaryLeaf(ImgSeqDataset):
    def __init__(self, root, phase, partition_file, label_file,transform=None,
                 ignore_index=None, label_ratio=-1, leaf_mask_ratio=-1, data_type = 'tiles'):
        self.label_ratio = label_ratio
        self.leaf_mask_ratio = leaf_mask_ratio
        self.data_type = data_type
        super(BinaryLeaf, self).__init__(root, phase, partition_file, label_file, transform, ignore_index)
        print(self.msg)

    def collect_samples(self):
        samples = []
        
        partition = namespace_to_dict(read_json(self.partition_file))[self.phase][self.data_type]
        for (_id, t) in partition.items():
            if  self.data_type == 'images':
                samples += [t] 
            elif 'leaf_mask_ratio' in t.keys():
                    if t['leaf_mask_ratio'] > self.leaf_mask_ratio and t['label_ratio'] > self.label_ratio:
                        samples += [t]
            elif t['label_ratio'] > self.label_ratio:
                        samples += [t]

                
        if  self.data_type == 'tiles':
            self.msg += f'Warning: BinaryLeaf: {self.phase} phase:\n' \
                        f'  loaded {len(samples)} out of ({len(partition)}) tiles from {self.root};\n'\
                        f'  label_ratio > {self.label_ratio}; leaf_ratio_mask > {self.leaf_mask_ratio};\n'
        else:
            self.msg += f'Warning: BinaryLeaf: {self.phase} phase:\n' \
                        f'  loaded {len(samples)} out of ({len(partition)}) images from {self.root};\n'
        return samples

    def set_labels(self):
        labels = read_json(self.label_file)
        label_ids = set()
        label_names = set()
        for species in labels.keys():
            for (idx, item) in labels[species].items():
                label_ids.add(item['number'])
                label_names.add(item['label'])
        self.msg += f'  label_names: {list(label_names)};\n'
        return labels

#     def set_ignore_index(self, batch):
#         assert batch['leaf_mask'].dtype == np.bool_ or batch['leaf_mask'].dtype == torch.bool, 'leaf_mask should be binary.'
#         if self.ignore_index:
#             batch['label'][~batch['leaf_mask']] = self.ignore_index
#         return batch

def test():
    path = '../../configs/config.yml'
    conf = read_from_yaml(path)
    d = conf.data
    trans = transforms.Compose(get_transforms(conf.training.transform))
    leafdata = BinaryLeaf(d.root[0], 'training', d.partition_file[0], d.label_file)
    
    list_idx = random.choices(range(len(leafdata)), k=5)
    for idx in list_idx:
        batch = leafdata.__getitem__(idx)
        import IPython; IPython.embed();
        exit()
        print(batch)
#         imgs = [convert_Tensor_to_numpy_img(it) for it in list(batch.values())]
#         plt_samples(imgs, n_r=1, n_c=3)


if __name__ == '__main__':
    test()
