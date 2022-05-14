import collections

import torch


def recall(tp, fn, esp=1e-6):
    return tp / (tp + fn + esp)


def precision(tp, fp, esp=1e-6):
    return tp / (tp + fp + esp)


def f_beta(tp, fp, fn, beta, esp=1e-6):
    p = precision(tp, fp, esp)
    r = recall(tp, fn, esp)
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r + esp)


def IoU(tp, fp, fn, esp=1e-6):
    return tp / (tp + fp + fn + esp)


def update_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update_dict(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = val
    return orig_dict


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor,mask=None):
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    if mask is not None:
        y_pred[~mask] = 0
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    tp, fp, tn, fn = confusion(y_pred, y_true)
    return tp, tn, fp, fn


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
#     # print(ignore_index)
#     if ignore_index is not None:
#         valid_index = (truth == ignore_index)
#         if is_remove_ignore_index:
#             prediction = prediction[~valid_index]
#             truth = truth[~valid_index]
#         else:
#             prediction[valid_index] = 0.0

    confusion_vector = prediction / truth
    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    return true_positives, false_positives, true_negatives, false_negatives


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3, names=None):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)
        self.names = names
        if names is not None:
            assert self.meters == len(self.names)
        else:
            self.names = [''] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert (len(val) == self.meters and len(n) == self.meters)
        for i in range(self.meters):
            self.count[i] += n[i]
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        val = ' '.join(['{} {:.{}f}'.format(n, v, self.precision) for n, v in
                        zip(self.names, self.val)])
        avg = ' '.join(['{} {:.{}f}'.format(n, a, self.precision) for n, a in
                        zip(self.names, self.avg)])
        return '{} ({})'.format(val, avg)
