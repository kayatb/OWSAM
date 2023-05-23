"""
Copied and adapted from: https://github.com/hysts/pytorch_mixup/blob/master/utils.py
"""

from utils.misc import labels_to_onehot

import torch
import torch.nn.functional as F
import numpy as np


def mixup(predictions, targets, alpha, n_classes, num_masks, pad_num):
    indices = torch.randperm(predictions.size(0))
    predictions2 = predictions[indices]
    targets2 = targets[indices]

    targets = labels_to_onehot(targets, n_classes)
    targets2 = labels_to_onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)]).to(predictions.device)
    predictions = predictions * lam + predictions2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return predictions, targets


def mixup_cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class MixUpCrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return mixup_cross_entropy_loss(input, target, self.size_average)
