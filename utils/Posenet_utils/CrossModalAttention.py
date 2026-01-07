import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super().__init__()
        
        # MODIFICA 1: L'input ora ha 4 canali in più (RGB+Depth+BBox)
        # channels*2 (1024) + 4 (bbox) = 1028
        self.input_dim = (channels * 2) + 4
        
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(self.input_dim, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Spatial Gate resta uguale (lavora sull'output ridotto)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel Gate deve vedere anche il BBox per capire il contesto di scala
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.input_dim, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, kernel_size=1), # Output scaling solo per RGB+Depth
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, depth_feat, bb_info):
        """
        rgb_feat: [B, 512, 7, 7]
        depth_feat: [B, 512, 7, 7]
        bb_info: [B, 4] -> Va espanso
        """
        B, _, H, W = rgb_feat.shape
        
        # 1. Espandiamo il BBox per farlo diventare una mappa spaziale [B, 4, 7, 7]
        # (Esattamente come fai nel modello principale)
        bb_spatial = bb_info.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # 2. Concateniamo TUTTO: RGB + Depth + Posizione
        combined = torch.cat([rgb_feat, depth_feat, bb_spatial], dim=1) # [B, 1028, 7, 7]
        
        # --- A. SPATIAL ATTENTION (Guidata ora anche dalla posizione!) ---
        x_compressed = self.channel_reducer(combined)
        spatial_mask = self.spatial_gate(x_compressed) 
        
        # --- B. CHANNEL ATTENTION ---
        channel_scale = self.channel_gate(combined)
        rgb_scale, depth_scale = torch.split(channel_scale, rgb_feat.size(1), dim=1)
        
        # --- C. APPLICAZIONE (Uguale a prima) ---
        rgb_attended = rgb_feat * spatial_mask
        depth_attended = depth_feat * spatial_mask
        
        rgb_final = rgb_attended * rgb_scale
        depth_final = depth_attended * depth_scale
        
        return rgb_final, depth_final, spatial_mask