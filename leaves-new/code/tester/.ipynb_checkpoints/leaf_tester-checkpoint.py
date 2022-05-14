import os

import numpy as np
import torch
from tqdm import tqdm
from path import Path
from tool.data_io import read_json, read_image, save_image
from tester.base_tester import BaseTester
from transforms.transform import Resize
from tool.misc_utils import confusion_matrix
class VisualizeFramework(BaseTester):
    """
    1. map {img_id:batch_indices}

    case1: single_target_name
    case2: all images
    """

    def __init__(self, loader, model, _log, cfg, dataset):
        super(VisualizeFramework, self).__init__(loader, model, _log, cfg)
        self.dataset = dataset
        self.partition = read_json(cfg.data.partition_file[0])[self.cfg.evaluate.phase]
        self.target_img_id = cfg.evaluate.single_target_img_id
        self.mask_name = self.cfg.evaluate.mask.name
        self.conf_eval = self.cfg.evaluate
        self.tile_root = self.conf_eval.save.tile.root
        self.image_root = self.conf_eval.save.image.root
        self.data_type = self.cfg.data.data_type
        self.id2batchIdx = self.get_id2batchIdx()

    def evaluate_one_image(self, target_img_id):
        species, base_name = target_img_id.split('/')
        img_act = img_pred = None

        if self.image_root:
            info = self.partition['images'][target_img_id]['info']
            shape_ = (info['height'], info['width'])
            resize = Resize(info['height'], info['width'])
            if self.conf_eval.save.is_activitions:
                img_act = np.zeros(shape_)
            if self.conf_eval.save.is_preds:
                img_pred = np.zeros(shape_)
        scores = np.array([0,0,0,0])
        # find all batches for the given img_id 
        for idx in self.id2batchIdx[target_img_id]:
            batch = self.dataset.get_batch(idx)
            if self.data_type == "tiles":
                tile_id = batch['id']
                r1, r2, c1, c2 = [int(v) for v in tile_id.split('-')[-4:]]
                t_shape = int(r2-r1), int(c2-c1)
            else:
                t_shape = int(self.cfg.training.transform.resize.m), int(self.cfg.training.transform.resize.n)
            
            imgs, masks = batch['image'].to(self.device, dtype=torch.float32), \
                          batch[self.mask_name].to(self.device, dtype=torch.bool)
            labels = batch['label'].to(self.device,dtype=torch.float32)
            logits = self.model(imgs).reshape(t_shape)
            logits[~masks.reshape(logits.shape)] = -1e6 #predict ~mask as negative
            prods = torch.sigmoid(logits)
            preds = (prods >= 0.5).bool()
            tp, tn, fp, fn = confusion_matrix(y_true=labels.flatten(), y_pred=preds.float().flatten(),mask=masks.flatten())
            scores += np.array([tp, tn, fp, fn])
            
            logits = logits.cpu().detach().numpy()
            preds = preds.reshape(t_shape).cpu().detach().numpy()
            if self.image_root and self.data_type == "tiles":
                if img_act is not None:
                    img_act[r1:r2, c1:c2] = logits
                if img_pred is not None:
                    img_pred[r1:r2, c1:c2] = preds
                    
            # This is to save image training on image level   
            if self.image_root and self.data_type == "images": 
                if img_act is not None:
                    img_act= resize(logits)
                if img_pred is not None:
                    img_pred = resize(preds)
                    
            # save to /tile_root/species/file_type/base_name/tile_id .png or .npy
            if self.tile_root:
                file_type = self.conf_eval.save.tile.type_name
                dir_ = Path(self.tile_root)/'tile'/self.cfg.evaluate.phase/ species / file_type / base_name
                dir_.makedirs_p()
                file_name = tile_id.split('/')[1]
                if self.conf_eval.save.is_activitions:
                    np.save(str(dir_ / (file_name +'.npy')), logits)
                if self.conf_eval.save.is_preds:
                    save_image(dir_ / file_name +'.png', preds)

        # save to /img_root/species/file_type/base_name .png or .npy
        if self.image_root:
            file_type = self.conf_eval.save.image.type_name
            dir_ = Path(self.image_root) /'image'/self.cfg.evaluate.phase/  species / file_type
            dir_.makedirs_p()
            if img_act is not None:
                np.save(dir_/base_name+'.npy', img_act)
            if img_pred is not None:
                save_image(dir_/base_name+'.png', img_pred.astype(bool))
        return scores

    def evaluate(self):
        if self.target_img_id and self.target_img_id not in self.id2batchIdx.keys():
            self._log.info(f'=>not found target_img_id')
            return
        scores = np.array([0,0,0,0])
        for img_id in tqdm(self.partition['images'].keys()):
            if self.target_img_id is None or self.target_img_id==img_id:
                scores += self.evaluate_one_image(img_id)
        return scores

                
    def get_id2batchIdx(self):
        id2batchIdx = {}
        for batch in tqdm(self.loader):
            idx = batch['idx'].item()  
            img_id = '_'.join(batch['id'][0].split('_')[:-1]) if self.data_type == "tiles" else batch['id'][0]
            id2batchIdx[img_id] = [idx] if img_id not in id2batchIdx.keys() else id2batchIdx[img_id] + [idx]
#             import IPython;
#             IPython.embed();
#             exit();

        return id2batchIdx
