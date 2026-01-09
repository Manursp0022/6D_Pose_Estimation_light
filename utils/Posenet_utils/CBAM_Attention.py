import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Shared MLP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compressione lungo i canali: Max e Avg
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Convoluzione Spaziale
        x_out = self.conv1(x_cat)
        
        # Mappa di Attenzione Spaziale (La "Maschera")
        scale = self.sigmoid(x_out) 
        return scale

class CBAM_CrossModal(nn.Module):
    def __init__(self, rgb_channels=512, depth_channels=512):
        super(CBAM_CrossModal, self).__init__()
        
        # Totale canali quando concateniamo
        total_channels = rgb_channels + depth_channels 
        
        # 1. Channel Attention (Capisce quali feature sono importanti tra RGB e Depth)
        self.ca = ChannelAttention(total_channels)
        
        # 2. Spatial Attention (Capisce DOVE è l'oggetto)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, rgb, depth, bb_info=None):
        # Concateniamo RGB e Depth
        x = torch.cat([rgb, depth], dim=1) # [B, 1024, 7, 7]
        
        # --- PHASE 1: Channel Refinement ---
        # "Quale modalità è più affidabile qui? RGB o Depth?"
        x = x * self.ca(x)
        
        # --- PHASE 2: Spatial Refinement ---
        # "Dove è l'oggetto?"
        # CBAM applica la maschera spaziale a tutto il pacchetto
        spatial_mask = self.sa(x)
        x = x * spatial_mask
        
        # Separiamo di nuovo per compatibilità con il resto della rete
        rgb_new, depth_new = torch.split(x, rgb.size(1), dim=1)
        
        # Ritorniamo anche la spatial_mask per il debug (la visualizzazione rossa)
        return rgb_new, depth_new, spatial_mask