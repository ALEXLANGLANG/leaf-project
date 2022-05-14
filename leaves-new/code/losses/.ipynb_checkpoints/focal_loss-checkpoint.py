
import torch
import torch.nn as nn


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:

        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6


    'Focal_Loss(p_t) = -alpha_t* (1 - p_t)^{\gamma}* log(p_t)'

    def forward(self, logits: torch.Tensor, label: torch.Tensor, mask=None) -> torch.Tensor:
        assert logits.shape == label.shape, f'logits.shape{logits.shape} != label.shape{logits.shape}'
        if mask is not None:
            assert mask.dtype == torch.bool, 'leaf_mask should be binary.'
            mask = mask.reshape(logits.shape)
            logits = logits[mask]
            label = label[mask]

        p_1 = torch.sigmoid(logits)
        p_1 = p_1.clamp(self.eps, 1. - self.eps)
        p_0 = 1.0 - p_1
        loss_1 = - label * self.alpha * torch.pow(p_0, self.gamma) * torch.log(p_1)
        loss_0 = - (1.0 - label) * (1 - self.alpha) * torch.pow(p_1, self.gamma) * torch.log(p_0)
        loss = loss_1 + loss_0

#         debug_print(p_1, p_0, loss_1, loss_0, label, logits, loss)

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def main():
    pass
#     criterion = BinaryFocalLossWithLogits(alpha=0.935, gamma=2.0, reduction='mean')
#     import numpy as np
#     preds = torch.Tensor([-np.nan, -1000, -.002, 0, 10000, 1.7, 0])
#     target = torch.Tensor([ignore_index, ignore_index, 0, 1, 0, 1, ignore_index])
#     preds = torch.reshape(preds, (1, 1, 1, len(preds)))
#     target = torch.reshape(target, (1, 1, 1, len(target)))
#     # loss = criterion(preds, target)
#     loss_tmp = criterion(preds, target)
#     # print(loss.item())
#     print(loss_tmp.item())


if __name__ == '__main__':
    main()

    
# def save_dict(logits, label, p_1, p_0, loss_1, loss_0):
#     torch.save(logits.detach().cpu(), './logits.pt')
#     torch.save(label.detach().cpu(), './label.pt')
#     torch.save(p_1.detach().cpu(), './p_1.pt')
#     torch.save(p_0.detach().cpu(), 'p_0.pt')
#     torch.save(loss_1.detach().cpu(), './loss_1.pt')
#     torch.save(loss_0.detach().cpu(), './loss_0.pt')
#     pass


# def debug_print(p_1, p_0, loss_1, loss_0, label, logits, loss):
#     if torch.any(torch.isnan(p_1)):
#         print('*********************DEBUGGING************************')
#         print(f'p_1 has nan: {torch.isnan(p_1).sum()}')
#     if torch.any(torch.isnan(p_0)):
#         print('*********************DEBUGGING************************')
#         print(f'p_0 has nan: {torch.isnan(p_0).sum()}')
#     if torch.any(torch.isnan(loss_1)):
#         print('*********************DEBUGGING************************')
#         print(f'loss_1 has nan: {torch.isnan(loss_1).sum()}')
#     if torch.any(torch.isnan(loss_0)):
#         print('*********************DEBUGGING************************')
#         print(f'loss_0 has nan: {torch.isnan(loss_0).sum()}')
#     if torch.any(torch.isnan(label)):
#         print('*********************DEBUGGING************************')
#         print(f'label has nan: {torch.isnan(label).sum()}')
#     if torch.any(torch.isnan(logits)):
#         print('*********************DEBUGGING************************')
#         print(f'logits has nan: {torch.isnan(logits).sum()}')
#     if torch.any(torch.isnan(loss)):
#         print('*********************DEBUGGING************************')
#         print(f'loss has nan: {torch.isnan(loss).sum()}')
#         save_dict(logits, label, p_1, p_0, loss_1, loss_0)
