import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionNet(nn.Module):
    def __init__(self, num_classes=13):
        super(PositionNet, self).__init__()
        
        # input_dim = 9 (intrinsics) + 4 (box: x,y,w,h) + num_classes (one-hot)
        input_dim = 9 + 4 + num_classes
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024), 
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.fc4 = nn.Linear(512, 3)  # Output: x, y, z

    def forward(self, x):
        result = self.fc1(x)
        result = self.fc2(result)
        result = self.fc3(result)
        result = self.fc4(result)
        return result