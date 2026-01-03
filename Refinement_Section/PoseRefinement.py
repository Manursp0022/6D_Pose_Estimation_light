import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseRefinement(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        # Input: features globali + pose iniziale (4 rot + 3 trans)
        self.refine_net = nn.Sequential(
            nn.Linear(feat_dim + 7, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # Delta: 4 quat + 3 trans
        )
    
    def forward(self, global_feat, pred_rot, pred_trans):
        # Concatena features + pose corrente
        pose_vec = torch.cat([pred_rot, pred_trans], dim=1)  # [B, 7]
        x = torch.cat([global_feat, pose_vec], dim=1)  # [B, 512+7]
        
        # Predici delta
        delta = self.refine_net(x)
        delta_rot = delta[:, :4]
        delta_trans = delta[:, 4:]
        
        # Applica correzione
        new_rot = F.normalize(pred_rot + delta_rot, dim=1)
        new_trans = pred_trans + delta_trans
        
        return new_rot, new_trans