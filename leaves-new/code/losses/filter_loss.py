import torch.nn as nn
import torch


def choose_hinge(hinge):
    if hinge == 'ReLU':
        res = nn.ReLU()
    elif hinge == 'LeakyReLU':
        res = nn.LeakyReLU()
    else:
        print(f'please select hing from (ReLU, LeakyReLU)')
        exit(1)
    print(f'FilerLoss: choose_hinge: {hinge}')
    return res


class FilterLossWithLogits(nn.Module):
    def __init__(self, a0: float = 0.1, a1: float = None, b1: float = 1, beta: float = 2, margin=1.0,
                 reduction: str = 'none',
                 eps=1e-6, hinge='ReLU'):
        super(FilterLossWithLogits, self).__init__()
        self.a0 = a0
        self.a1 = 1. - a0 if a1 is None else a1
        self.b1 = b1
        self.beta = beta
        self.eps = eps
        self.margin = margin
        self.hinge = choose_hinge(hinge)
        self.reduction = reduction


    def forward(self, logits: torch.Tensor, label: torch.Tensor,mask=None) -> torch.Tensor:
        assert logits.shape == label.shape, f'logits.shape{logits.shape} != label.shape{logits.shape}'
        if mask is not None:
            assert mask.dtype == torch.bool, 'leaf_mask should be binary.'
            mask = mask.reshape(logits.shape)
            logits = logits[mask]
            label = label[mask]
        loss0 = (1.0 - label) * self.a0 * self.hinge(logits + self.margin)
        loss1 = self.hinge(-logits + self.margin)
        loss1 = label * (self.a1 * loss1 + self.b1 * (torch.pow(loss1 + 1, self.beta) - 1 ))
        msg = f'loss0:{loss0}\n' \
              f'loss1:{loss1}'
        # print(msg)
        loss = loss0 + loss1
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def main():
    criterion = FilterLossWithLogits(a0=0.01, b1=1, beta=2)
    import numpy as np
    logits = torch.Tensor([-2,-1,-0.5,0,0.5,1,2])
    target1 = torch.Tensor([1]*7)
    target0 = torch.Tensor([0]*7)
    l=torch.Tensor([-2,-1,-0.5,0,0.5,1,2,-2,-1,-0.5,0,0.5,1,2])
    t = torch.Tensor([1]*7+[0]*7)
#     preds = torch.reshape(preds, (1, 1, 1, len(preds)))
#     target = torch.reshape(target, (1, 1, 1, len(target)))
    loss_1 = criterion(logits, target1)
    loss_0 = criterion(logits, target0)
    loss_ = criterion(l, t)
    print(loss_1)
    print(loss_0)
    print(loss_)


if __name__ == '__main__':
    main()
