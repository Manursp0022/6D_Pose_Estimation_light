import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyPinholeRefiner(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        
        
        self.class_bias = nn.Embedding(16, 3)  # Per-class bias
        nn.init.zeros_(self.class_bias.weight)  # Start from zero correction
        
        # Tiny MLP for bbox + pinhole features
        self.refiner = nn.Sequential(
            nn.Linear(7, 32),  # 3 (xyz) + 4 (bbox)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
        # Output constraint (CRITICAL!)
        self.max_delta = 0.15  # Max ±15cm correction

    def forward(self, pinhole_xyz, bbox, class_ids):
        # Normalize inputs (IMPORTANT!)
        xyz_norm = pinhole_xyz / torch.tensor([1.0, 1.0, 1.0]).to(pinhole_xyz.device)
        
        # Concatenate features
        features = torch.cat([xyz_norm, bbox], dim=1)
        
        # Predict delta
        delta = self.refiner(features)
        
        # Add class-specific bias
        class_correction = self.class_bias(class_ids)
        delta = delta + class_correction
        
        # Constrain to reasonable range
        delta = torch.tanh(delta) * self.max_delta
        
        return pinhole_xyz + delta