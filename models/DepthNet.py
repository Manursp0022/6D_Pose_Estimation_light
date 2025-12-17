import torch
import torch.nn as nn
from torchvision import models

class DepthNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthNet, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
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