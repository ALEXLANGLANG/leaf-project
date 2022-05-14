import torch

from tool.torch_utils import load_checkpoint
from abc import abstractmethod


class BaseTester:
    """

    """

    def __init__(self, loader, model, _log, cfg):
        self._log = _log
        self.loader = loader
        self.cfg = cfg
        self.device, self.device_ids = self._prepare_device(self.cfg.training.hardware.num_gpu)
        self.model = self._init_model(model)
        

    @abstractmethod
    def evaluate(self):
        pass

    # double check data parallel and load_state
    def _init_model(self, model):
        cfg_m = self.cfg.model
        model = model.to(self.device)
        if cfg_m.load_from:
            self._log.info("=> using pre-trained weights {}.".format(
                cfg_m.load_from))
            epoch, weights = load_checkpoint(cfg_m.load_from, self.device)
            model.load_state_dict(weights)
        else:
#             self._log.info("=> Please specify the model {}.".format(
#                 cfg_m.load_from))
#             raise NotImplementedError
            self._log.info("=> Use init model.")
                        

        model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        self._log.info("number of parameters: {}".format(self.count_parameters(model)))
        self._log.info("gpu memory allocated (model parameters only): {} Bytes".format(torch.cuda.memory_allocated()))
        return model

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