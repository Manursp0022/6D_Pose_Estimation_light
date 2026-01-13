import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 

class DenseFusion_Masked_DualAtt_NetVar(nn.Module):
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
        
        self.feat_dropout = nn.Dropout2d(p=0.3)
        self.head_dropout = nn.Dropout2d(p=0.15)
        
        #FUSION with RESIDUAL
        self.fusion_entry = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.fusion_res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            self.head_dropout,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512)
        )

        self.rot_head = nn.Sequential(
            nn.Conv2d(1024, 256, 1), 
            nn.ReLU(),
            nn.Conv2d(256, 4, 1)    
        )
        
        self.trans_head = nn.Sequential(
            nn.Conv2d(520, 256, 1), 
            nn.ReLU(),
            nn.Conv2d(256, 3, 1)    
        )
        
        self.conf_head = nn.Sequential(
            nn.Conv2d(1032, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1) 
        )
    
    def _forward_fusion(self, rgb, depth, mask=None):
        if mask is not None:
            rgb = rgb * mask
            depth = depth * mask
        
        if rgb.is_cuda:
            # Create streams for parallel execution
            rgb_stream = torch.cuda.Stream()
            depth_stream = torch.cuda.Stream()
            
            # Wait for current stream
            rgb_stream.wait_stream(torch.cuda.current_stream())
            depth_stream.wait_stream(torch.cuda.current_stream())
            
            with torch.cuda.stream(rgb_stream):
                rgb_feat = self.rgb_extractor(rgb)       
                rgb_feat = self.feat_dropout(rgb_feat)

            with torch.cuda.stream(depth_stream):
                depth_3ch = torch.cat([depth, depth, depth], dim=1)
                depth_feat = self.depth_extractor(depth_3ch) 
                depth_feat = self.feat_dropout(depth_feat)
            
            # Synchronize streams with current stream
            torch.cuda.current_stream().wait_stream(rgb_stream)
            torch.cuda.current_stream().wait_stream(depth_stream)
        else:
            # Fallback for CPU
            rgb_feat = self.rgb_extractor(rgb)       
            depth_3ch = torch.cat([depth, depth, depth], dim=1)
            depth_feat = self.depth_extractor(depth_3ch) 

            rgb_feat = self.feat_dropout(rgb_feat)
            depth_feat = self.feat_dropout(depth_feat)

        rgb_enhanced = rgb_feat 
        depth_enhanced = depth_feat 
        
        # FUSION + RESIDUAL
        combined = torch.cat([rgb_enhanced, depth_enhanced], dim=1)
        x = self.fusion_entry(combined)
        x_res = self.fusion_res(x)
        fused_feat = F.relu(x + x_res) # Residual add & final ReLU
        
        return fused_feat, rgb_enhanced, depth_enhanced

    def _weighted_pooling(self, fused_feat, batch_size, rgb_enhanced, depth_enhanced, bb_info, cam_params):
        """Logica di pooling intelligente condivisa"""

        rot_input = torch.cat([fused_feat, rgb_enhanced], dim=1)
        pred_rot_map = self.rot_head(rot_input)     # [B, 4, 7, 7]

        bb_spatial = bb_info.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)      # [B, 4, 7, 7]
        cam_spatial = cam_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)  # [B, 4, 7, 7]

        trans_input = torch.cat([fused_feat, bb_spatial, cam_spatial], dim=1) 
        pred_trans_map = self.trans_head(trans_input) # [B, 3, 7, 7]

        conf_input = torch.cat([fused_feat, rgb_enhanced, bb_spatial, cam_spatial], dim=1) 
        conf_logits = self.conf_head(conf_input)   # [B, 1, 7, 7] (Logits)
        
        pred_rot_map = pred_rot_map.view(batch_size, 4, -1)   # [B, 4, 49]
        pred_trans_map = pred_trans_map.view(batch_size, 3, -1) # [B, 3, 49] Ha senso ancora questa .view 
        conf_logits = conf_logits.view(batch_size, 1, -1)     # [B, 1, 49]
        
        pred_rot_map = F.normalize(pred_rot_map + self.eps, p=2, dim=1)
        
        # Softmax with Temperature onlogits
        weights = F.softmax(conf_logits / self.temperature, dim=2) #[B, 1, 49]
        
        # Weighted Average
        pred_rot_global = torch.sum(pred_rot_map * weights, dim=2)   #[B, 4]
        pred_trans_global = torch.sum(pred_trans_map * weights, dim=2) # [B, 3]
        
        # Final Normalize
        pred_rot_global = F.normalize(pred_rot_global + self.eps, p=2, dim=1)

        return pred_rot_global, pred_trans_global

    def forward(self, rgb, depth,bb_info, cam_params, mask=None):
        bs = rgb.size(0)
        fused_feat, rgb_enhanced, depth_enhanced = self._forward_fusion(rgb, depth, mask)
        pred_r, pred_t = self._weighted_pooling(fused_feat, bs, rgb_enhanced, depth_enhanced,  bb_info, cam_params)

        return pred_r, pred_t

