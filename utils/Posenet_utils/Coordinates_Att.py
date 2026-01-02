import torch
import torch.nn as nn

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, in_channels // reduction)
        
        self.conv_reduce = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU()
        
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1)
        
        # Proiezione finale a 1 canale per compatibilità
        self.proj = nn.Conv2d(in_channels, 1, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.relu(self.bn(self.conv_reduce(y)))
        
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        att_h = torch.sigmoid(self.conv_h(x_h))
        att_w = torch.sigmoid(self.conv_w(x_w))
        
        att = att_h * att_w  # [B, C, H, W]
        
        return torch.sigmoid(self.proj(att))  # [B, 1, H, W]