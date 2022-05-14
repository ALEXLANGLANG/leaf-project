import argparse
import os
from datetime import datetime
from shutil import copyfile

from path import Path
from easydict import EasyDict
import pprint


import sys

import basic_train
from tool.yaml_io import read_from_yaml, namespace_to_dict

sys.path.append('../')
from tool.data_io import read_json
from tool.logger import init_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument('-r', '--resume', default=None)
    parser.add_argument('-n', '--name', default='')
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()
    if args.resume is not None:
        args.resume = Path(args.resume)
        args.config = args.resume / 'config.yml'
        

    cfg = EasyDict(namespace_to_dict(read_from_yaml(args.config)))
    if args.resume is not None:
        cfg.resume = args.resume
    # cfg.train.DEBUG = args.DEBUG

    if args.DEBUG:
        cfg.training.size.update({
            'epoch_num': 20,
            'epoch_size': 2,
            'val_epoch_size': 1,
            'valid_size': 2
        })
        cfg.save_root = Path(cfg.save_root)/'debug'
        
    if args.evaluate:
        cfg.training.size.update({
            'epoch_size': -1,
            'valid_size':0,
            'workers':1,
            'val_epoch_size':1
        })

        
    if args.model is not None:
        cfg.training.model.load_from = args.model
    # store files day by day
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume is None:
        args.resume = cfg.resume
    if args.resume is not None:
        cfg.save_root = args.resume
        _log = init_logger(log_dir=cfg.save_root, filename='resume' + curr_time + '.log')
    else:
        if args.name == '':
            args.name = os.path.basename(args.config)[:-5]
        cfg.save_root = Path(cfg.save_root) / args.name + "_" +  curr_time
        cfg.save_root.makedirs_p()
        copyfile(args.config, cfg.save_root / 'config.yml')
        _log = init_logger(log_dir=cfg.save_root, filename='train_' + curr_time + '.log')

    # init logger
    
    _log.info('=> slurm jobid: {}'.format(os.environ.get('SLURM_JOBID')))
    _log.info('will save everything to {}'.format(cfg.save_root))

    # show configurations
    cfg_str = pprint.pformat(cfg)
    _log.info('=> configurations \n ' + cfg_str)
    basic_train.main(cfg, _log, resume=args.resume is not None)

    pass


if __name__ == '__main__':
    main()
