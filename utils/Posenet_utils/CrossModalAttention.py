import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super().__init__()
        
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Spatial Attention Map Generator
        # Guarda il contesto locale (7x7) combinato
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, kernel_size=1), # Output: 1 canale (Mappa di importanza)
            nn.Sigmoid() # Mappa tra 0 e 1 (Soft Mask)
        )
        
        # 3. Channel Attention (Opzionale, ispirato ai paper SE-Net / CBAM)
        # Serve a dire "quali filtri sono importanti" (es. bordi verticali vs texture)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, depth_feat):
        # Concateniamo le feature grezze
        combined = torch.cat([rgb_feat, depth_feat], dim=1) # [B, 1024, 7, 7]
        
        # --- A. SPATIAL ATTENTION (La tua "Soft Mask") ---
        # "Dove guardare?"
        x_compressed = self.channel_reducer(combined)
        spatial_mask = self.spatial_gate(x_compressed) # [B, 1, 7, 7]
        
        # --- B. CHANNEL ATTENTION (Raffinamento) ---
        # "Cosa cercare?"
        channel_scale = self.channel_gate(combined) # [B, 1024, 1, 1]
        rgb_scale, depth_scale = torch.split(channel_scale, rgb_feat.size(1), dim=1)
        
        # --- C. FUSIONE CROSS-GATED ---
        # 1. Applichiamo la Soft Mask Spaziale a entrambi
        rgb_attended = rgb_feat * spatial_mask
        depth_attended = depth_feat * spatial_mask
        
        # 2. Applichiamo il Channel Scaling (Ricalibrazione)
        rgb_final = rgb_attended * rgb_scale
        depth_final = depth_attended * depth_scale
        
        return rgb_final, depth_final, spatial_mask