import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
#from utils.Posenet_utils.attention import GeometricAttention 

class DenseFusion_Masked_DualAtt_NetVarNoMask(nn.Module):
    def __init__(self, pretrained=True, temperature=2.0):
        super().__init__()
        
        self.temperature = temperature 
        self.eps = 1e-8                

        self.rgb_backbone = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.depth_backbone = models.resnet18(weights='DEFAULT' if pretrained else None)

        # list(... children()) takes all the pieces of ResNet. 
        #[:-2] Cut away the last two pieces: Global Average Pooling (which would flatten everything 
        #to 1 x 1) and the final Fully Connected Layer.
        self.rgb_extractor = nn.Sequential(*list(self.rgb_backbone.children())[:-2]) #
        self.depth_extractor = nn.Sequential(*list(self.depth_backbone.children())[:-2])
        
        self.attention_block = GeometricAttention(in_channels=512)
        #self.attention_block = CBAM_CrossModal(rgb_channels=512, depth_channels=512)

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
        
        # Confidence Head (Output Logits per Softmax)
        self.conf_head = nn.Sequential(
            nn.Conv2d(1032, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1) 
        )
    
    def _forward_fusion(self, rgb, depth, return_debug=True):
        
        rgb_feat = self.rgb_extractor(rgb)       
        depth_3ch = torch.cat([depth, depth, depth], dim=1)
        depth_feat = self.depth_extractor(depth_3ch) 

        rgb_feat = self.feat_dropout(rgb_feat)
        depth_feat = self.feat_dropout(depth_feat)

        #att_map = self.attention_block(depth_feat) 
        rgb_clean, depth_clean, soft_mask = self.attention_block(rgb_feat, depth_feat)
        
        # FUSION + RESIDUAL
        combined = torch.cat([rgb_clean, depth_clean], dim=1)
        x = self.fusion_entry(combined)
        x_res = self.fusion_res(x)
        fused_feat = F.relu(x + x_res) # Residual add & final ReLU
        
        debug_info = {}
        if return_debug:
            debug_info['attention_map'] = soft_mask
        
        return fused_feat, rgb_clean, depth_clean, debug_info

    def _weighted_pooling(self, fused_feat, batch_size, rgb_clean, depth_clean, bb_info,cam_params):
        """Logica di pooling intelligente condivisa"""

        rot_input = torch.cat([fused_feat, rgb_clean], dim=1)
        pred_rot_map = self.rot_head(rot_input)     # [B, 4, 7, 7]

        bb_spatial = bb_info.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)      # [B, 4, 7, 7]
        cam_spatial = cam_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)  # [B, 4, 7, 7]

        trans_input = torch.cat([fused_feat, bb_spatial, cam_spatial], dim=1) 
        pred_trans_map = self.trans_head(trans_input) # [B, 3, 7, 7]

        conf_input = torch.cat([fused_feat, rgb_clean, bb_spatial, cam_spatial], dim=1) 
        conf_logits = self.conf_head(conf_input)   # [B, 1, 7, 7] (Logits)
        
        # Flatten Spatial Dimensions
        pred_rot_map = pred_rot_map.view(batch_size, 4, -1)   # [B, 4, 49]
        pred_trans_map = pred_trans_map.view(batch_size, 3, -1) # [B, 3, 49] Ha senso ancora questa .view 
        conf_logits = conf_logits.view(batch_size, 1, -1)     # [B, 1, 49]
        
        # Normalize Quaternions (with Epsilon)
        pred_rot_map = F.normalize(pred_rot_map + self.eps, p=2, dim=1)
        
        # --- WEIGHT CALCULATION ---
        # Softmax with Temperature onlogits
        weights = F.softmax(conf_logits / self.temperature, dim=2) #[B, 1, 49]
        
        # Weighted Average
        pred_rot_global = torch.sum(pred_rot_map * weights, dim=2)   #[B, 4]
        pred_trans_global = torch.sum(pred_trans_map * weights, dim=2) # [B, 3]
        
        # Final Normalize
        pred_rot_global = F.normalize(pred_rot_global + self.eps, p=2, dim=1)

        return pred_rot_global, pred_trans_global

    def forward(self, rgb, depth, bb_info, cam_params, mask=None, return_debug=True):
        bs = rgb.size(0)
        fused_feat, rgb_enhanced, depth_enhanced, dbg = self._forward_fusion(rgb, depth, mask, return_debug=True)
        pred_r, pred_t = self._weighted_pooling(fused_feat, bs, rgb_enhanced, depth_enhanced,  bb_info, cam_params)
        
        if return_debug:
            return pred_r, pred_t, dbg
        return pred_r, pred_t

