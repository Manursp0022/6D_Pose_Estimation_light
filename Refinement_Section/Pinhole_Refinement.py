import torch
import torch.nn as nn
import torch.nn.functional as F

class PinholeRefineNet(nn.Module):
    def __init__(self):
        super(PinholeRefineNet, self).__init__()
        
        # Branch per coordinate e intrinseche (4 parametri + 3 coordinate)
        self.geo_feat = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Branch per Bounding Box (4 parametri: x1, y1, x2, y2)
        self.bbox_feat = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Testa di regressione finale
        self.regressor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3) # Output: Delta X, Delta Y, Delta Z
        )

    def forward(self, pinhole_xyz, intrinsics, bbox):
        # Flattening intrinsics and concatenation (16,4) intrinsics
        fx = intrinsics[:, 0].unsqueeze(1)  # Primo valore
        fy = intrinsics[:, 1].unsqueeze(1)  # Secondo valore
        cx = intrinsics[:, 2].unsqueeze(1)  # Terzo valore
        cy = intrinsics[:, 3].unsqueeze(1)  # Quarto valore

        #print(fx)
        geo_input = torch.cat([pinhole_xyz, fx, fy, cx, cy], dim=1)
        
        f_geo = self.geo_feat(geo_input)
        f_bbox = self.bbox_feat(bbox)
        
        combined = torch.cat([f_geo, f_bbox], dim=1)
        delta_xyz = self.regressor(combined)
        
        # Correzione finale
        refined_xyz = pinhole_xyz + delta_xyz
        return refined_xyz