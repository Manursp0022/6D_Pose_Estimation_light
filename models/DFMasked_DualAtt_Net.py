import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from utils.Posenet_utils.attention import GeometricAttention 

class DenseFusion_Masked_DualAtt_Net(nn.Module):
    def __init__(self, pretrained=True, temperature=1.0):
        super().__init__()
        
        self.temperature = temperature # Idea 2: Scaling
        self.eps = 1e-8                # Idea 1: Stabilità

        # --- 1. BACKBONES ---
        self.rgb_backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.depth_backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        
        # Output: 512 channels, 7x7 spatial
        self.rgb_extractor = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_extractor = nn.Sequential(*list(self.depth_backbone.children())[:-2])
        
        # --- 2. ATTENTION MODULE ---
        self.attention_block = GeometricAttention(in_channels=512)
        
        # --- 3. DENSE FUSION CON RESIDUAL ---
        # Input: 1024 -> Project to 512 -> Residual Block
        self.fusion_entry = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Residual Block interno alla fusione
        self.fusion_res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512)
        )
        
        # --- 4. PIXEL-WISE HEADS ---
        # Rot Head
        self.rot_head = nn.Sequential(
            nn.Conv2d(512, 256, 1), 
            nn.ReLU(),
            nn.Conv2d(256, 4, 1)    
        )
        
        # Trans Head
        self.trans_head = nn.Sequential(
            nn.Conv2d(520, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 3, 1)    
        )
        
        # Confidence Head (Output Logits per Softmax)
        self.conf_head = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1) 
        )

    def _forward_fusion(self, rgb, depth, mask=None, return_debug=False):
        """Helper function condivisa tra forward e refine"""
        # MASKING
        if mask is not None:
            print("-----> Mask found , Applying mask <-----")
            rgb = rgb * mask
            depth = depth * mask
        
        # EXTRACT
        rgb_feat = self.rgb_extractor(rgb)       
        depth_3ch = torch.cat([depth, depth, depth], dim=1)
        depth_feat = self.depth_extractor(depth_3ch) 
        
        # DUAL ATTENTION
        att_map = self.attention_block(depth_feat) 
        
        # RESIDUAL ATTENTION: (1 + att) per non perdere segnale
        rgb_enhanced = rgb_feat * (1 + att_map) 
        depth_enhanced = depth_feat * (1 + att_map) 
        #depth_for_trans = depth_feat
        
        # FUSION + RESIDUAL
        combined = torch.cat([rgb_enhanced, depth_enhanced], dim=1)
        x = self.fusion_entry(combined)
        # Residual connection: x + Block(x)
        x_res = self.fusion_res(x)
        fused_feat = F.relu(x + x_res) # Residual add & final ReLU
        
        debug_info = {}
        if return_debug:
            # Calcolo statistiche solo se richiesto
            debug_info['att_mean'] = att_map.mean().item()
            debug_info['att_max']  = att_map.max().item()
            debug_info['att_min']  = att_map.min().item()
            debug_info['att_std']  = att_map.std().item()
        
        return fused_feat, debug_info

    def _weighted_pooling(self, fused_feat, batch_size,bb_info,cam_params,return_debug=False, debug_info=None):
        """Logica di pooling intelligente condivisa"""

        bb_spatial = bb_info.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)      # [B, 4, 7, 7]
        cam_spatial = cam_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)  # [B, 4, 7, 7]

        pred_rot_map = self.rot_head(fused_feat)     # [B, 4, 7, 7]
        #fused_with_depth = torch.cat([fused_feat,depth_stats], dim=1)
        trans_input = torch.cat([fused_feat, bb_spatial, cam_spatial], dim=1)
        pred_trans_map = self.trans_head(trans_input) # [B, 3, 7, 7]
        conf_logits = self.conf_head(fused_feat)     # [B, 1, 7, 7] (Logits)
        
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
        fused_flat = fused_feat.view(batch_size, 512, -1)
        vector_feat_global = torch.sum(fused_flat * weights, dim=2) # [B, 512]
        
        if return_debug and debug_info is not None:
            debug_info['conf_max'] = weights.max().item()
            debug_info['conf_mean'] = weights.mean().item()
            debug_info['conf_std'] = weights.std().item()

        return pred_rot_global, pred_trans_global, vector_feat_global, debug_info

    def forward(self, rgb, depth,bb_info, cam_params, mask=None, return_debug=False):
        bs = rgb.size(0)
        fused_feat, dbg = self._forward_fusion(rgb, depth, mask, return_debug)
        pred_r, pred_t, _, dbg_final = self._weighted_pooling(fused_feat, bs, bb_info, cam_params, return_debug, dbg)
        
        if return_debug:
            return pred_r, pred_t, dbg_final
        return pred_r, pred_t

    def forward_refine(self, rgb, depth, bb_info, cam_params, mask=None):
        bs = rgb.size(0)
        fused_feat, _ = self._forward_fusion(rgb, depth, mask, return_debug=False)
        pred_r, pred_t, vector_feat, _ = self._weighted_pooling(
            fused_feat, bs, bb_info, cam_params, return_debug=False, debug_info=None
        )
        return pred_r, pred_t, vector_feat