import torch
import torch.nn as nn
from torchvision import models

class PoseResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(PoseResNet, self).__init__()
        
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        #The output size of the ResNet50 backbone is 2048.
        feature_dim = 2048
        
        # Regression Head 
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4)  # Quaternion (w, x, y, z)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        
        q = self.regressor(x)
        
        q = torch.nn.functional.normalize(q, p=2, dim=1)
        
        return q