from torch.utils.data import DataLoader
import sys

sys.path.append('../')
from datasets.get_dataset import get_dataset
from losses.get_loss import get_loss
from models.get_model import get_model
from trainer.get_trainer import get_trainer


def main(cfg, _log, resume=False):
    # 1. datasets
    # 2. loader
    # 3. model
    # 4. loss
    # 5. trainer
    _log.info("=> fetching img pairs.")
    train_set, valid_set = get_dataset(cfg)
    _log.info('{} samples found, {} train samples and {} {} samples '.format(
        len(valid_set) + len(train_set),
        len(train_set),
        len(valid_set), cfg.evaluate.phase))

    t = cfg.training
    val_loader = DataLoader(valid_set, batch_size=t.size.batch_size, shuffle=False, num_workers=t.hardware.workers,
                            pin_memory=True, drop_last=False)

    train_loader = DataLoader(train_set, batch_size=t.size.batch_size, shuffle=True,
                              num_workers=t.hardware.workers,
                              pin_memory=True)

    if cfg.training.size.epoch_size == 0:
        cfg.training.size.epoch_size = len(train_loader)
    if cfg.training.size.valid_size == 0:
        cfg.training.size.valid_size = len(val_loader)

    model = get_model(cfg.training.model)
    loss = get_loss(cfg)
    trainer = get_trainer(cfg)(train_loader, val_loader, model, loss, _log,
                               cfg.save_root, cfg, resume=resume)

    trainer.train()
