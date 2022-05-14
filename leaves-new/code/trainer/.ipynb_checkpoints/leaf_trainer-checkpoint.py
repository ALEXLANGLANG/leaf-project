import numpy as np
from torch import nn
import torch
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer
from tool.plt_utils import print_dict
from tool.misc_utils import precision, recall, f_beta, IoU, confusion_matrix
from tool.yaml_io import read_from_yaml


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, cfg, resume=False):
        self.train_loader = train_loader
        super(TrainFramework, self).__init__(train_loader, valid_loader, model, loss_func,
                                             _log, save_root, cfg, resume=resume)
        # self.ignore_index = cfg.data.ignore_index
        self.mask_name = self.cfg.training.mask.name
    def _run_one_epoch(self):
        # 1. load data
        # 2. forward, zero_grad, backward, step update
        # 5. evaluation
        self.model.train()
        loss_ep = 0
        cm_ep = np.array([0, 0, 0, 0])
        with tqdm(total=len(self.train_loader), desc=f'Epoch {self.i_epoch + 1}/{self.cfg.training.size.epoch_num}', unit='batch') as pbar:
            for i_step, batch in enumerate(self.train_loader):
                if i_step >= self.cfg.training.size.epoch_size:
                    break
                
                imgs, labels, masks = batch['image'].to(self.device, dtype=torch.float32), batch['label'].to(self.device,dtype=torch.float32), batch[self.mask_name].to(self.device,dtype=torch.bool)
                logits = self.model(imgs)
                loss = self.loss_func(logits=logits, label=labels,mask=masks)
                prods = torch.sigmoid(logits)
                preds = (prods >= 0.5).float()
                tp, tn, fp, fn = confusion_matrix(y_true=labels.flatten(), y_pred=preds.flatten(), mask=masks.flatten())
                loss_ep += loss.item()
                cm_ep += np.array([tp, tn, fp, fn])
                # compute gradients and do optimization step
                self.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                self.i_iter += 1
                pbar.update()
            self.i_epoch += 1

            # save info to log and tensorboard
            lr = self.optimizer.param_groups[0]['lr']
            tp, tn, fp, fn = cm_ep
            scalars = [self.i_epoch, loss_ep, lr, tp, tn, fp, fn, precision(tp, fp), recall(tp, fn), IoU(tp, fp, fn),
                       f_beta(tp, fp, fn, 1), f_beta(tp, fp, fn, 2)]
            scalars_names = ['epoch','loss', 'lr', 'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'IoU', 'f1', 'f2']
            self.save_scalars(scalars, scalars_names, 'training', self.i_epoch)
            self.save_log_info(scalars, scalars_names, 'training', self.i_epoch)

    @torch.no_grad()
    def _validate_with_gt(self):
        self.model.eval()
        cm_ep = np.array([0, 0, 0, 0])
        with tqdm(total=len(self.valid_loader), unit='batch') as pbar:
            for i_step, batch in enumerate(self.valid_loader):
                if i_step >= self.cfg.training.size.valid_size:
                    break
                imgs, labels = batch['image'].to(self.device, dtype=torch.float32), batch['label'].to(self.device,dtype=torch.float32)
                masks =  None if self.mask_name is None else batch[self.mask_name].to(self.device,dtype=torch.bool).flatten()
                logits = self.model(imgs)
                prods = torch.sigmoid(logits)
                preds = (prods >= 0.5).float()
                tp, tn, fp, fn = confusion_matrix(y_true=labels.flatten(), y_pred=preds.flatten(),mask=masks)
                cm_ep += np.array([tp, tn, fp, fn])
                pbar.update()

        # save info to log and tensorboard
        tp, tn, fp, fn = cm_ep
        scalars_names = ['tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'IoU', 'f1', 'f2', 'pre+rec']
        scalars = [tp, tn, fp, fn, precision(tp, fp), recall(tp, fn), IoU(tp, fp, fn),
                   f_beta(tp, fp, fn, 1), f_beta(tp, fp, fn, 2),precision(tp, fp)+recall(tp, fn)]
        self.save_scalars(scalars, scalars_names, 'validation', self.i_epoch)
        self.save_log_info(scalars, scalars_names, 'validation', self.i_epoch)

        cfg_save = self.cfg.training.save
        f_score = f_beta(tp, fp, fn, int(cfg_save.save_F_beta_score))
        self.save_model(f_score, name=f'model', type_ ='max')
        if self.i_epoch % cfg_save.save_model_every_epoch==0:
            self.save_model(f_score, name=f'model_ep{self.i_epoch}', is_best=False)


def test():
    cfg = read_from_yaml('../../../configs/config.yml')
    msg = print_dict(cfg.training)
    print(msg)
    import pprint
    cfg_str = pprint.pformat(cfg)
    print(cfg_str)


if __name__ == '__main__':
    test()
