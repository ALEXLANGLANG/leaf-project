import sys
sys.path.append('../')
import argparse
from datetime import datetime
from easydict import EasyDict
from path import Path
import os
import pprint
import numpy as np
from tool.yaml_io import namespace_to_dict,read_from_yaml
from tool.logger import init_logger
from tool.misc_utils import f_beta, precision, recall
from torch.utils.data import DataLoader
from tester.leaf_tester import VisualizeFramework
from datasets.get_dataset import get_dataset
from models.get_model import get_model

def visualize(cfg, _log):
    _log.info(f"=> start to visualize predictions on {cfg.evaluate.phase}.")
    _log.info("=> fetching img pairs.")
    train_set, valid_set = get_dataset(cfg)
    val_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=False)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False,
                              num_workers=4,
                              pin_memory=True)
    dataset = train_set if cfg.evaluate.phase =='training' else valid_set
    loader = train_loader if cfg.evaluate.phase =='training' else val_loader
    model = get_model(cfg.model)
        
    v = VisualizeFramework(loader, model, _log, cfg, dataset)
    return v.evaluate()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('--DEBUG', action='store_true')
    args = parser.parse_args()
    cfg = EasyDict(namespace_to_dict(read_from_yaml(args.config)))
    if args.DEBUG:
        cfg.evaluate.update({
            'single_target_img_id': 'quercus-bicolor-herbivory/00000750_2144617',
            'phase': 'validation'
        })
        cfg.evaluate.save.update({
            'is_activitions': True,
            'is_preds': True
        })
        cfg.evaluate.save.image.update({
            'root': '/usr/xtmp/xs75/leaves/tmp/visual_img',
            'type_name': 'tmp_mask'
        })
        cfg.evaluate.save.tile.update({
            'root': '/usr/xtmp/xs75/leaves/tmp/visual_tile',
            'type_name': 'tmp_mask'
        })
        cfg.model.load_from = '/usr/xtmp/xs75/leaves/exps/preAugF09_20211106_151637/model_model_best.pth.tar'


    if args.model is not None:
        cfg.training.model.load_from = args.model
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # init logger
    cfg.evaluate.save.log_root = Path(cfg.evaluate.save.log_root)
    cfg.evaluate.save.log_root.makedirs_p()
    _log = init_logger(log_dir=cfg.evaluate.save.log_root, filename='visual' + curr_time + '.log')
    _log.info('=> slurm jobid: {}'.format(os.environ.get('SLURM_JOBID')))
    if cfg.evaluate.save.image.root:
        _log.info('will save images to {}'.format(cfg.evaluate.save.image.root))
    if cfg.evaluate.save.tile.root:
        _log.info('will save tiles to {}'.format(cfg.evaluate.save.tile.root))
            # show configurations
    cfg_str = pprint.pformat(cfg)
    _log.info('=> configurations \n ' + cfg_str)
    
    phases = cfg.evaluate.phase
    scores = []
    
    tot_tp = tot_tn = tot_fp = tot_fn = 0
    tot_score = np.array([0,0,0,0])
    for phase in phases:
        cfg.evaluate.phase = phase
        tp, tn, fp, fn = visualize(cfg, _log)
        tot_score += np.array([tp, tn, fp, fn])
        scores.append([tp, tn, fp, fn, recall(tp, fn), precision(tp, fp), f_beta(tp, fp, fn, 1), f_beta(tp, fp, fn, 2)])
        
    tot_tp, tot_tn, tot_fp, tot_fn = tot_score
    scores.append([tot_tp, tot_tn, tot_fp, tot_fn, recall(tot_tp, tot_fn), precision(tot_tp, tot_fp), 
                   f_beta(tot_tp, tot_fp, tot_fn, 1), f_beta(tot_tp, tot_fp, tot_fn, 2)])
    
<<<<<<< HEAD
    _log.info('=> phases: \n', phases)
    _log.info('=> scores:  tp, tn, fp, fn, recall, precision, f1, f2' )    
    _log.info('=> scores: \n ', scores)
=======
#     _log.info('=> phases: \n', phases)
#     _log.info('=> scores:  tp, tn, fp, fn, recall, precision, f1, f2' )    
#     _log.info('=> scores: \n ', scores)
>>>>>>> new-name

if __name__=='__main__':
    main()