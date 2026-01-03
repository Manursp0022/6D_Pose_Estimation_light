import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from utils.Posenet_utils.attention import GeometricAttention 

class DenseFusion_Masked_DualAtt_NetVarGlobal_WRef(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.eps = 1e-8                

        self.rgb_backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.depth_backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        
        self.rgb_extractor = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_extractor = nn.Sequential(*list(self.depth_backbone.children())[:-2])
        
        self.attention_block = GeometricAttention(in_channels=512)

        self.feat_dropout = nn.Dropout2d(p=0.3)
        self.head_dropout = nn.Dropout2d(p=0.15)
        
        #FUSION with RESIDUAL
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
            self.head_dropout,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512)
        )
        
        # Pixel wise heads
        # Rot Head
        self.rot_head = nn.Sequential(
            nn.Conv2d(1024, 256, 1), 
            nn.ReLU(),
            nn.Conv2d(256, 4, 1)    
        )
        
        # Trans Head
        self.trans_head = nn.Sequential(
            nn.Conv2d(520, 256, 1), 
            nn.ReLU(),
            nn.Conv2d(256, 3, 1)    
        )

    
    def _forward_fusion(self, rgb, depth, mask=None, return_debug=False):
        """Helper function condivisa tra forward e refine"""
        # MASKING
        if mask is not None:
            rgb = rgb * mask
            depth = depth * mask
        
        # EXTRACT
        rgb_feat = self.rgb_extractor(rgb)       
        depth_3ch = torch.cat([depth, depth, depth], dim=1)
        depth_feat = self.depth_extractor(depth_3ch) 

        #Adding this to avoid overfitting on hard versione
        rgb_feat = self.feat_dropout(rgb_feat)
        depth_feat = self.feat_dropout(depth_feat)
        
        # DUAL ATTENTION
        att_map = self.attention_block(depth_feat) 
        #att_map_depth = self.attention_block(depth_feat)
        #att_map_rgb = self.attention_block(rgb_feat)

        # RESIDUAL ATTENTION: (1 + att) per non perdere segnale
        rgb_enhanced = rgb_feat * (1 + att_map)  
        depth_enhanced = depth_feat * (1 + att_map) #[B, 512, 7, 7]
        
        # FUSION + RESIDUAL
        combined = torch.cat([rgb_enhanced, depth_enhanced], dim=1)
        x = self.fusion_entry(combined)
        # Residual connection: x + Block(x)
        x_res = self.fusion_res(x)
        fused_feat = F.relu(x + x_res) # Residual add & final ReLU
        
        debug_info = {}
        if return_debug:
            debug_info['att_max_rgb']  = att_map.max().item()
            debug_info['att_min_rgb']  = att_map.min().item()
            debug_info['att_std_rgb']  = att_map.std().item()
        
        return fused_feat, rgb_enhanced, depth_enhanced, debug_info

    def _global_pooling(self, fused_feat, batch_size, rgb_enhanced, depth_enhanced, bb_info,cam_params,return_debug=False, debug_info=None):

        rot_input = torch.cat([fused_feat, rgb_enhanced], dim=1)
        pred_rot_map = self.rot_head(rot_input)     # [B, 4, 7, 7]

        bb_spatial = bb_info.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)      # [B, 4, 7, 7]
        cam_spatial = cam_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)  # [B, 4, 7, 7]
        trans_input = torch.cat([fused_feat, bb_spatial, cam_spatial], dim=1) 
        pred_trans_map = self.trans_head(trans_input) # [B, 3, 7, 7]

        # GLOBAL AVERAGE POOLING (invece di weighted)
        pred_rot_global = pred_rot_map.mean(dim=[2, 3])     # [B, 4]
        pred_trans_global = pred_trans_map.mean(dim=[2, 3]) # [B, 3]
        
        # Final Normalize
        pred_rot_global = F.normalize(pred_rot_global + self.eps, p=2, dim=1)
        
        if return_debug and debug_info is not None:
            debug_info['conf_max'] = weights.max().item()
            debug_info['conf_mean'] = weights.mean().item()
            debug_info['conf_std'] = weights.std().item()

        return pred_rot_global, pred_trans_global, debug_info

    def forward(self, rgb, depth,bb_info, cam_params, num_refine=2, mask=None, return_debug=False):
        bs = rgb.size(0)
        fused_feat, rgb_enhanced, depth_enhanced, dbg = self._forward_fusion(rgb, depth, mask, return_debug)
        pred_r, pred_t, dbg_final = self._global_pooling(fused_feat, bs, rgb_enhanced, depth_enhanced,  bb_info, cam_params, return_debug, dbg)
        
        global_feat = fused_feat.mean(dim=[2, 3])
        # Iterative refinement
        for _ in range(num_refine):
            pred_rot, pred_trans = self.refinement(global_feat_weighted, pred_r, pred_t)

        if return_debug:
            return pred_r, pred_t, dbg_final
        return pred_r, pred_t

