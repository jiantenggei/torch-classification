
import torch
import torch.nn as nn
import torch.nn.functional as F
# version 1: use torch.autograd
class FocalLoss(nn.Module):
    #------------------------------------------#
    #   FocalLoss ：处理样本不均衡
    #   alpha   
    #   gamma >0 当 gamma=0 时就是交叉熵损失函数
    #   论文中gamma = [0,0.5,1,2,5]
    #   一般而言当γ增加的时候，a需要减小一点
    #   reduction ： 就平均：'mean' 求和 'sum'
    #------------------------------------------#
    def __init__(self,alpha=0.25,gamma=2,reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, logits, label):
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss