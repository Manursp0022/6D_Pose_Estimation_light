import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from utils.Posenet_utils.attention import GeometricAttention 

class DenseFusion_Masked_DualAtt_Net34(nn.Module):
    def __init__(self, pretrained=True, temperature=1.0):
        super().__init__()

        self.temperature = temperature
        self.eps = 1e-8

        # --- 1. UPGRADE BACKBONES: ResNet34 + DILATION ---
        # ResNet34 ha più capacità della 18.
        # replace_stride_with_dilation=[False, False, True] mantiene la risoluzione finale a 14x14 invece di 7x7
# --- 1. RESNET-50 (Necessaria per la Dilation) ---
        self.rgb_backbone = models.resnet50(
            weights='DEFAULT' if pretrained else None,
            replace_stride_with_dilation=[False, False, True] # Output 14x14
        )
        self.depth_backbone = models.resnet50(
            weights='DEFAULT' if pretrained else None,
            replace_stride_with_dilation=[False, False, True]
        )

        # ResNet34 output channels (layer4) è 512, come ResNet18, quindi il resto non cambia dimensione
        self.rgb_extractor = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_extractor = nn.Sequential(*list(self.depth_backbone.children())[:-2])
        
        # --- 2. ATTENTION ---
        self.attention_block = GeometricAttention(in_channels=512)
        
        # --- 3. FUSION (Più profonda) ---
        self.fusion_entry = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        self.fusion_res = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512)
        )
        
        # --- 4. HEADS DECOUPLED (Separate e più profonde) ---
        # Rotazione (più complessa)
        self.rot_head = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), # CORRETTO: 1024 input
            nn.ReLU(),
            nn.Conv2d(512, 256, 1), 
            nn.ReLU(),
            nn.Conv2d(256, 4, 1)    
        )
        
        # Traslazione
        self.trans_head = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), # CORRETTO: 1024 input
            nn.ReLU(),
            nn.Conv2d(512, 256, 1), 
            nn.ReLU(),
            nn.Conv2d(256, 3, 1)    
        )
        
        # Confidence
        self.conf_head = nn.Sequential(
            nn.Conv2d(512, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 1, 1) 
        )

    def _forward_fusion(self, rgb, depth, mask=None, return_debug=False):
        if mask is not None:
            rgb = rgb * mask
            rgb = rgb * mask
            depth = depth * mask
        
        # Extract & Project
        rgb_feat = self.rgb_extractor(rgb) 
        rgb_feat = self.project_rgb(rgb_feat) # Include Dropout
        
        depth_3ch = torch.cat([depth, depth, depth], dim=1)
        depth_feat = self.depth_extractor(depth_3ch)
        depth_feat = self.project_depth(depth_feat) # Include Dropout
        
        # Attention
        att_map = self.attention_block(depth_feat) 
        rgb_enhanced = rgb_feat * (1 + att_map) 
        depth_enhanced = depth_feat * (1 + att_map) 
        
        # Fusion
        combined = torch.cat([rgb_enhanced, depth_enhanced], dim=1)
        x = self.fusion_entry(combined)
        x_res = self.fusion_res(x)
        fused_feat = F.relu(x + x_res)
        
        return fused_feat, rgb_enhanced, depth_enhanced, None

    def _weighted_pooling(self, fused_feat, rgb_enhanced, depth_enhanced, batch_size):
        rot_input = torch.cat([fused_feat, rgb_enhanced], dim=1)
        trans_input = torch.cat([fused_feat, depth_enhanced], dim=1)
        
        pred_rot_map = self.rot_head(rot_input)
        pred_trans_map = self.trans_head(trans_input)
        conf_logits = self.conf_head(fused_feat)
        
        pred_rot_map = pred_rot_map.view(batch_size, 4, -1)   
        pred_trans_map = pred_trans_map.view(batch_size, 3, -1) 
        conf_logits = conf_logits.view(batch_size, 1, -1)     
        
        pred_rot_map = F.normalize(pred_rot_map + self.eps, p=2, dim=1)
        weights = F.softmax(conf_logits / self.temperature, dim=2) 
        
        pred_rot_global = torch.sum(pred_rot_map * weights, dim=2)   
        pred_trans_global = torch.sum(pred_trans_map * weights, dim=2) 
        pred_rot_global = F.normalize(pred_rot_global + self.eps, p=2, dim=1)
        
        return pred_rot_global, pred_trans_global

    def forward(self, rgb, depth, mask=None, return_debug=False):
        bs = rgb.size(0)
        fused, rgb_enh, depth_enh, dbg = self._forward_fusion(rgb, depth, mask, return_debug)
        pred_r, pred_t = self._weighted_pooling(fused, rgb_enh, depth_enh, bs)
        return pred_r, pred_t