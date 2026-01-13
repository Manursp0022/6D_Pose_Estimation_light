import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 

class DenseFusion_NetVar(nn.Module):
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
        
        # Residual Block 
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

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, rgb, depth, bb_info, cam_params):
            bs = rgb.size(0)
            
            rgb_feat = self.rgb_extractor(rgb)       
            depth_3ch = torch.cat([depth, depth, depth], dim=1)
            depth_feat = self.depth_extractor(depth_3ch) 

            rgb_feat = self.feat_dropout(rgb_feat)
            depth_feat = self.feat_dropout(depth_feat)
            
            combined = torch.cat([rgb_feat, depth_feat], dim=1)
            x_f = self.fusion_entry(combined)
            x_res = self.fusion_res(x_f)
            fused_feat = F.relu(x_f + x_res)
            
            # Spatial extension
            bb_spatial = bb_info.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)
            cam_spatial = cam_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)

            #Features injection + predictions
            rot_input = torch.cat([fused_feat, rgb_feat], dim=1)
            pred_r_map = self.rot_head(rot_input)
            pred_r = self.global_pool(pred_r_map).view(bs, 4)
            pred_r = F.normalize(pred_r + self.eps, p=2, dim=1)

            trans_input = torch.cat([fused_feat, bb_spatial, cam_spatial], dim=1)
            pred_trans_map = self.trans_head(trans_input)
            pred_t = self.global_pool(pred_trans_map).view(bs, 3)

            return pred_r, pred_t