import torch
import torch.nn as nn
import torchvision.models as models

class GeometricAttention(nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        self.conv_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1), 
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.conv_att(x)