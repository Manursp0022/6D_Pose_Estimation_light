import torch
from torch import Tensor
from torch.nn import functional as F

class weightedTranslationLoss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(weightedTranslationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, weight: Tensor = None) -> Tensor:
        loss = F.mse_loss(input, target, reduction='none')

        if weight is not None:
            loss = loss * weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss