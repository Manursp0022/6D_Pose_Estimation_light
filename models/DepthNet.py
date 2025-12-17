import torch
import torch.nn as nn
from torchvision import models

class DepthNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthNet, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        original_conv = resnet.conv1 
        
        # 2. Creiamo un nuovo layer con 1 solo canale in ingresso
        # Manteniamo invariati out_channels, kernel_size, stride, padding e bias
        resnet.conv1 = nn.Conv2d(1, original_conv.out_channels, 
                                 kernel_size=original_conv.kernel_size, 
                                 stride=original_conv.stride, 
                                 padding=original_conv.padding, 
                                 bias=original_conv.bias)

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        #The output size of the ResNet18 backbone is 512.
        feature_dim = 512
        
        # Regression Head 
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)  # Depth value
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)

        d = self.regressor(x)

        return d