import torch
import torch.nn as nn
import torchvision.models as models 
from utils.Posenet_utils.attention import GeometricAttention 

class DenseFusion_RGBD_Net(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.rgb_backbone = models.resnet18(pretrained=pretrained)
        self.depth_backbone = models.resnet18(pretrained=pretrained)
        
        # Removing final FC's -> Output: 512 channels, 7x7 spatial
        self.rgb_extractor = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_extractor = nn.Sequential(*list(self.depth_backbone.children())[:-2])
        
        #Geometric Attention
        self.attention_block = GeometricAttention(in_channels=512)
        
        #Fusion block:
        # Instead of flattening immediately, we process the merged features (RGB+Depth)
        # Input: 1024 (512 RGB + 512 Depth) -> Output: 512
        self.fusion_net = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Let's maintain the 7x7 spatiality until the last moment
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # PREDICTION HEADS
        
        # Rotation
        self.rot_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Aggiunto dropout per sicurezza
            nn.Linear(256, 4) 
        )
        
        # Translation
        self.trans_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3) 
        )
        
        """
        # Confidence
        self.conf_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        """

    def forward(self, rgb, depth):
        #Feature Extraction
        rgb_feat = self.rgb_extractor(rgb)       # [B, 512, 7, 7]
        
        depth_3ch = torch.cat([depth, depth, depth], dim=1)
        depth_feat = self.depth_extractor(depth_3ch) # [B, 512, 7, 7]
        
        # Depth “tells” RGB where to look, we produce an attention map
        att_map = self.attention_block(depth_feat) 
        rgb_enhanced = rgb_feat * att_map 
        
        # Concatenazione and Fusion
        combined_feat = torch.cat([rgb_enhanced, depth_feat], dim=1) # [B, 1024, 7, 7]
        
        fused_feat = self.fusion_net(combined_feat) # [B, 512, 7, 7]
        
        #Pooling & Heads
        vector_feat = self.global_pool(fused_feat) 
        vector_feat = torch.flatten(vector_feat, 1) # [B, 512]
        
        pred_rot = self.rot_head(vector_feat)
        pred_rot = torch.nn.functional.normalize(pred_rot, p=2, dim=1)
        
        pred_trans = self.trans_head(vector_feat)
        #pred_conf = self.conf_head(vector_feat)
        
        return pred_rot, pred_trans
        #pred_conf