import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from utils.Posenet_utils.attention import GeometricAttention 

class DenseFusion_Masked_DualAtt_NetVarV2(nn.Module):
    def __init__(self, pretrained=True, temperature=2.0):
        super().__init__()
        
        self.temperature = temperature # Idea 2: Scaling
        self.eps = 1e-8                # Idea 1: Stabilità

        # BACKBON
        #self.rgb_backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        #self.depth_backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.shared_extractor = nn.Sequential(*list(backbone.children())[:-2])

        self.depth_adapter = nn.Conv2d(1, 3, kernel_size=1)
        
        # ATTENTION MODULE ---
        self.attention_block = GeometricAttention(in_channels=512)

        self.feat_dropout = nn.Dropout2d(p=0.3)
        self.head_dropout = nn.Dropout2d(p=0.15)
        
        # --- 3. DENSE FUSION  ---
        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.15)
        )
        
        # --- 4. PIXEL-WISE HEADS ---
        self.rot_head = nn.Sequential(
            nn.Conv2d(1024, 512, 1), 
            nn.ReLU(), 
            self.head_dropout,
            nn.Conv2d(512, 128, 1), 
            nn.ReLU(),
            nn.Conv2d(128, 4, 1)
        )
        
        # Input per trans: 512 + 4 + 4 = 520
        self.trans_head = nn.Sequential(
            nn.Conv2d(520, 256, 1), 
            nn.ReLU(), 
            self.head_dropout,
            nn.Conv2d(256, 128, 1), 
            nn.ReLU(),
            nn.Conv2d(128, 3, 1)
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(1032, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1) 
        )

    def _forward_fusion(self, rgb, depth, mask=None, return_debug=False):
        """Helper function condivisa tra forward e refine"""
        # MASKING
        if mask is not None:
            rgb = rgb * mask
            depth = depth * mask


        # Feature extraction (backbone condiviso)
        rgb_feat = self.shared_extractor(rgb)
        depth_3ch = self.depth_adapter(depth)
        depth_feat = self.shared_extractor(depth_3ch)

        # Dropout
        rgb_feat = self.feat_dropout(rgb_feat)
        depth_feat = self.feat_dropout(depth_feat)
        
        # DUAL ATTENTION
        att_map = self.attention_block(depth_feat) 
        
        # RESIDUAL ATTENTION: (1 + att) per non perdere segnale
        rgb_enhanced = rgb_feat * (1 + att_map) 
        depth_enhanced = depth_feat * (1 + att_map) 

        fused = self.fusion(torch.cat([rgb_enhanced, depth_enhanced], dim=1))
        
        debug_info = {}
        if return_debug:
            # Calcolo statistiche solo se richiesto
            debug_info['att_mean'] = att_map.mean().item()
            debug_info['att_max']  = att_map.max().item()
            debug_info['att_min']  = att_map.min().item()
            debug_info['att_std']  = att_map.std().item()
        
        return fused, rgb_enhanced, debug_info

    def _weighted_pooling(self, fused_feat, batch_size, rgb_enhanced, bb_info,cam_params,return_debug=False, debug_info=None):
        """Logica di pooling intelligente condivisa"""

        rot_input = torch.cat([fused_feat, rgb_enhanced], dim=1)
        pred_rot_map = self.rot_head(rot_input)     # [B, 4, 7, 7]


        bb_spatial = bb_info.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)      # [B, 4, 7, 7]
        cam_spatial = cam_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)  # [B, 4, 7, 7]
        trans_input = torch.cat([fused_feat, bb_spatial, cam_spatial], dim=1)
        pred_trans_map = self.trans_head(trans_input) # [B, 3, 7, 7]

        conf_input = torch.cat([fused_feat, rgb_enhanced, bb_spatial, cam_spatial], dim=1)
        conf_logits = self.conf_head(conf_input)   # [B, 1, 7, 7] (Logits)
        
        # Flatten Spatial Dimensions
        pred_rot_map = pred_rot_map.view(batch_size, 4, -1)   # [B, 4, 49]
        pred_trans_map = pred_trans_map.view(batch_size, 3, -1) # [B, 3, 49] Ha senso ancora questa . view 
        conf_logits = conf_logits.view(batch_size, 1, -1)     # [B, 1, 49]
        
        # Normalize Quaternions (with Epsilon)
        pred_rot_map = F.normalize(pred_rot_map + self.eps, p=2, dim=1)
        
        # --- WEIGHT CALCULATION ---
        # Softmax con Temperatura sui logits
        weights = F.softmax(conf_logits / self.temperature, dim=2) # [B, 1, 49]
        
        # Weighted Average
        pred_rot_global = torch.sum(pred_rot_map * weights, dim=2)   # [B, 4]
        pred_trans_global = torch.sum(pred_trans_map * weights, dim=2) # [B, 3]
        # Final Normalize
        pred_rot_global = F.normalize(pred_rot_global + self.eps, p=2, dim=1)
        
        # Global Feature Vector (Weighted)
        #fused_flat = fused_feat.view(batch_size, 512, -1)
        #vector_feat_global = torch.sum(fused_flat * weights, dim=2) # [B, 512]
        
        if return_debug and debug_info is not None:
            debug_info['conf_max'] = weights.max().item()
            debug_info['conf_mean'] = weights.mean().item()
            debug_info['conf_std'] = weights.std().item()

        return pred_rot_global, pred_trans_global, debug_info

    def forward(self, rgb, depth,bb_info, cam_params, mask=None, return_debug=False):
        bs = rgb.size(0)
        fused_feat, rgb_enhanced, dbg = self._forward_fusion(rgb, depth, mask, return_debug)
        pred_r, pred_t, dbg_final = self._weighted_pooling(fused_feat, bs, rgb_enhanced, bb_info, cam_params, return_debug, dbg)
        
        if return_debug:
            return pred_r, pred_t, dbg_final
        return pred_r, pred_t