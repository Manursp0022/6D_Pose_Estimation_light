"""
But be careful: the distance between quaternions is not a simple subtraction.
A quaternion q and its opposite -q represent the exact same rotation. Loss must take this into account.
"""
import torch
from torch import nn

class QuaternionLoss(nn.Module):
    def __init__(self):
        super(QuaternionLoss, self).__init__()

    def forward(self, pred_q, gt_q):
        # ensure normalization
        pred_q = torch.nn.functional.normalize(pred_q, p=2, dim=1)
        gt_q = torch.nn.functional.normalize(gt_q, p=2, dim=1)
        
        # Scalar product (cosine similarity)
        # q1 . q2 measures how close the angles are
        dot_product = torch.sum(pred_q * gt_q, dim=1)
        
        # Loss = 1 - |q1 . q2|
        # We use the absolute value because q and -q are the same rotation.       
        loss = 1.0 - torch.abs(dot_product)
        
        return loss.mean()