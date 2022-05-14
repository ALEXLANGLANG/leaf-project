import torch
import numpy as np
from abc import abstractmethod
from path import Path
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from tool.torch_utils import load_checkpoint, bias_parameters, weight_parameters, save_checkpoint


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, cfg, resume=False):
        self._log = _log
        self.cfg = cfg
        self.save_root = Path(save_root)
        self.summary_writer = SummaryWriter(str(save_root))
        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.device, self.device_ids = self._prepare_device(self.cfg.training.hardware.num_gpu)

        if not resume:
            self.model = self._init_model(model)
            self.optimizer = self._create_optim()
            self.scheduler = self._create_scheduler()
            self.best_error = np.inf
            self.i_epoch = 0
            self.i_iter = 0
        else:
            self._load_resume_ckpt(model)
        self.loss_func = loss_func

    @abstractmethod
    def _run_one_epoch(self):
        pass

    @abstractmethod
    def _validate_with_gt(self):
        pass

    def train(self):
        cfg_size = self.cfg.training.size
        for epoch in range(self.i_epoch, cfg_size.epoch_num):
            self._run_one_epoch()
            if epoch % cfg_size.val_epoch_size == 0:
                self._validate_with_gt()
            self.scheduler.step()

    def zero_grad(self):
        # One Pytorch tutorial suggests clearing the gradients this way for faster speed
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        for param in self.model.parameters():
            param.grad = None

    # double check data parallel and load_state
    def _init_model(self, model):
        cfg_m = self.cfg.training.model
        model = model.to(self.device)
        if cfg_m.load_from:
            self._log.info("=> using pre-trained weights {}.".format(
                cfg_m.load_from))
            epoch, weights = load_checkpoint(cfg_m.load_from,self.device)
            # from collections import OrderedDict
            # new_weights = OrderedDict()
            # model_keys = list(model.state_dict().keys())
            # weight_keys = list(weights.keys())
            # for a, b in zip(model_keys, weight_keys):
            #     new_weights[a] = weights[b]
            # weights = new_weights
            model.load_state_dict(weights)
        else:
            self._log.info("=> Train from scratch.")
            # model.init_weights()
        model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        self._log.info("number of parameters: {}".format(self.count_parameters(model)))
        self._log.info("gpu memory allocated (model parameters only): {} Bytes".format(torch.cuda.memory_allocated()))
        return model

    def _create_optim(self):
        cfg_op = self.cfg.training.optim
        params = [
            {'params': bias_parameters(self.model.module),
             'weight_decay': cfg_op.bias_decay},
            {'params': weight_parameters(self.model.module),
             'weight_decay': cfg_op.weight_decay}]
        print(f'=> choose_optim: optimizer is {cfg_op.name}')
        params =self.model.parameters()
        if cfg_op.name == 'Adam':
            return optim.Adam(params, lr=cfg_op.lr, betas=(cfg_op.momentum, cfg_op.beta))
        elif cfg_op.name == 'SGD':
            return optim.SGD(params, lr=cfg_op.lr, momentum=cfg_op.momentum)
        elif cfg_op.name == 'AdamW':
            # import IPython; IPython.embed()
            # exit()
            return optim.AdamW(params, lr=cfg_op.lr, betas=(cfg_op.momentum, cfg_op.beta))
        else:
            raise NotImplementedError(cfg_op.name)

    def _create_scheduler(self):
        cfg_sch = self.cfg.training.scheduler
        print(f'=> choose_scheduler: optimizer is {cfg_sch.name}')
        if cfg_sch.name == 'MultiStepLR':
            return optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=cfg_sch.milestones,
                                                  gamma=cfg_sch.lr_reduction)
        # elif self.cfg.training.scheduler.name == 'OneCycleLR':
        #     # Warning: the training process cannot be resumed if the program is shut down by accident
        #     epochs = int(t.epochs - t.init_epoch)
        #     last_epoch = -1 if t.init_epoch == 0 else int(t.init_epoch * t.batch_size)  # This is to resume the training
        #     return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=t.learning_rate * 10,
        #                                          steps_per_epoch=len(data_loader),
        #                                          epochs=epochs, last_epoch=last_epoch)
        else:
            raise NotImplementedError(cfg_sch.name)

    def _load_resume_ckpt(self, model):
        self._log.info('==> resuming')
        ckpt_dict = torch.load(self.save_root /'model_ckpt.pth.tar')
        model = model.to(self.device)
        model.load_state_dict(ckpt_dict['state_dict'])
        self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        self.optimizer = self._create_optim()
        self.scheduler = self._create_scheduler()
        if 'optimizer_dict' in ckpt_dict.keys():
            self.optimizer.load_state_dict(ckpt_dict['optimizer_dict'])
        if 'scheduler_dict' in ckpt_dict.keys():
            self.scheduler.load_state_dict(ckpt_dict['scheduler_dict'])
            # Since we first save checkpoints and then do scheduler.step(),
            # the saved scheduler state is one step behind, so we make it up here
            self.scheduler.step()
        if 'iter' not in ckpt_dict.keys():
            self.i_iter = ckpt_dict['epoch'] * self.cfg.training.size.epoch_size
        else:
            self.i_iter = ckpt_dict['iter']
        if 'best_error' not in ckpt_dict.keys():
            self.best_error = np.inf
        self.i_epoch = ckpt_dict['epoch']

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self._log.warning("Warning: There\'s no GPU available on this machine,"
                              "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self._log.warning(
                "Warning: The number of GPU\'s configured to use is {}, "
                "but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        self._log.info('=> gpu in use: {} gpu(s)'.format(n_gpu_use))
        self._log.info('device names: {}'.format([torch.cuda.get_device_name(i) for i in list_ids]))
        return device, list_ids

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_model(self, error, name, is_best=None, type_='min'):

        if is_best is None:
            is_best = error < self.best_error if type_ == 'min' else error > self.best_error

        if is_best:
            self.best_error = error

        models = {'epoch': self.i_epoch,
                  'iter': self.i_iter,
                  'best_metric': self.best_error,
                  'state_dict': self.model.module.state_dict(),
                  'optimizer_dict': self.optimizer.state_dict(),
                  'scheduler_dict': self.scheduler.state_dict()}

        save_checkpoint(self.save_root, models, name, is_best)

    def save_scalars(self, scalars, scalars_names, phase, time_stamp):
        for v, n in zip(scalars, scalars_names):
            self.summary_writer.add_scalar(f'{phase}/{n}', v, time_stamp)

    def save_log_info(self, scalars, scalars_names, phase, time_stamp):
        log_msg = f'Epoch{time_stamp}: '
        for v, n in zip(scalars, scalars_names):
            log_msg += f'{phase}/{n}:{v:.3f} |'
        self._log.info(log_msg)
