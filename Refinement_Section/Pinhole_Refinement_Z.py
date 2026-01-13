import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ImageBasedTranslationNet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()
        
        # Camera intrinsics: [fx, cx, fy, cy]
        self.register_buffer('intrinsics', 
                    torch.tensor([0.8994, 0.5082, 1.1949, 0.5043], 
                                dtype=torch.float32))
        
        # Visual feature extractor
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final FC and avgpool
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Optional: freeze early layers for faster training
        for param in list(self.feature_extractor.parameters())[:20]:
            param.requires_grad = False
        
        # Geometric encoder: bbox (4) + intrinsics (4) = 8 dims
        self.geo_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64), # Normalize geometric path
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Fusion layer: visual features + geometric features
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + 128, 512),
            nn.BatchNorm1d(512), # Critical: Normalize the combined features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1) # Output: Z (meters)
        )
        
    def forward(self, image, bbox):
        """
        Args:
            image: (B, 3, H, W) - RGB image
            bbox: (B, 4) - normalized bbox [x_center, y_center, width, height]
        
        Returns:
            depth: (B, 1) - predicted Z in meters
        """
        batch_size = image.size(0)
        
        # Expand intrinsics to match batch size
        intrinsics_batch = self.intrinsics.unsqueeze(0).expand(batch_size, -1)  # (B, 4)
        
        # Extract visual features
        visual_feat = self.feature_extractor(image)  # (B, 512, H', W')
        visual_feat = F.adaptive_avg_pool2d(visual_feat, 1)  # (B, 512, 1, 1)
        visual_feat = visual_feat.view(batch_size, -1)  # (B, 512)

        # Concatenate all geometric features: bbox (4) + intrinsics (4) = 8
        geo_input = torch.cat([bbox, intrinsics_batch], dim=1)  # (B, 8)
        geo_feat = self.geo_encoder(geo_input)  # (B, 128)
        
        # Fuse visual and geometric features
        combined = torch.cat([visual_feat, geo_feat], dim=1)  # (B, 512 + 128)
        Z = self.fusion(combined)  # (B, 1)
        
        return Z