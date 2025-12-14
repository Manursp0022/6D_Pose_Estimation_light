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
        # Assicuriamoci che siano normalizzati (per sicurezza)
        pred_q = torch.nn.functional.normalize(pred_q, p=2, dim=1)
        gt_q = torch.nn.functional.normalize(gt_q, p=2, dim=1)
        
        # Prodotto scalare (cosine similarity)
        # q1 . q2 misura quanto sono vicini gli angoli
        dot_product = torch.sum(pred_q * gt_q, dim=1)
        
        # Loss = 1 - |q1 . q2|
        # Usiamo il valore assoluto perché q e -q sono la stessa rotazione
        loss = 1.0 - torch.abs(dot_product)
        
        return loss.mean()