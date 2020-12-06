from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights


@LOSSES.register_module()
class CEL_Sigmoid(nn.Module):
    def __init__(self, use_sample_weight=False, size_average=True):
        super(CEL_Sigmoid, self).__init__()

        self.use_sample_weight = use_sample_weight
        self.size_average = size_average

    def forward(self, logits, targets, sample_weight):
        batch_size = logits.shape[0]

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.use_sample_weight:
            weight = ratio2weight(targets_mask, sample_weight)
            loss = (loss * weight.cuda())

        loss = loss.sum() / batch_size if self.size_average else loss.sum()
        loss = dict(CEL_Sigmoid_loss=loss)

        return loss
