
import logging
import os
import random
import numpy as np
import torch.nn.functional as F
import torch

#------------------------------
# smooth_labels loss 函数
#------------------------------
def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean') # KL 散度
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)



def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    device = true_labels.device
    true_labels = torch.nn.functional.one_hot(
        true_labels, classes).detach().cpu()
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(
            size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)

        true_dist.scatter_(1, torch.LongTensor(
            index.unsqueeze(1)), confidence)
    return true_dist.to(device)