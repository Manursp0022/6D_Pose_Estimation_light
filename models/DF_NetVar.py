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
        # --- NUOVE TESTE DI PREDIZIONE ---
        
        # Rot Head: Fused + RGB (512 + 512 = 1024)
        self.rot_head = nn.Sequential(
            nn.Conv2d(1024, 256, 1), 
            nn.ReLU(),
            nn.Conv2d(256, 4, 1)    
        )
        
        # XY Head: Fused + BB + Cam (512 + 4 + 4 = 520)
        self.xy_head = nn.Sequential(
            nn.Conv2d(520, 128, 1), 
            nn.ReLU(),
            nn.Conv2d(128, 2, 1)    
        )
        
        # Z Head: Fused + Depth (512 + 512 = 1024)
        self.z_head = nn.Sequential(
            nn.Conv2d(1024, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, rgb, depth, bb_info, cam_params):
            bs = rgb.size(0)
            
            # 1. Estrazione feature
            rgb_feat = self.rgb_extractor(rgb)       
            depth_3ch = torch.cat([depth, depth, depth], dim=1)
            depth_feat = self.depth_extractor(depth_3ch) 

            rgb_feat = self.feat_dropout(rgb_feat)
            depth_feat = self.feat_dropout(depth_feat)
            
            # 2. Fusione Residuale
            combined = torch.cat([rgb_feat, depth_feat], dim=1)
            x_f = self.fusion_entry(combined)
            x_res = self.fusion_res(x_f)
            fused_feat = F.relu(x_f + x_res)
            
            # 3. Preparazione input spaziali per le teste
            bb_spatial = bb_info.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)
            cam_spatial = cam_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)

            # 4. Esecuzione Teste
            # Rotazione (Fused + RGB)
            rot_input = torch.cat([fused_feat, rgb_feat], dim=1)
            pred_r_map = self.rot_head(rot_input)
            pred_r = self.global_pool(pred_r_map).view(bs, 4)
            pred_r = F.normalize(pred_r + self.eps, p=2, dim=1)

            # XY (Fused + BB + Cam)
            xy_input = torch.cat([fused_feat, bb_spatial, cam_spatial], dim=1)
            pred_xy_map = self.xy_head(xy_input)
            pred_xy = self.global_pool(pred_xy_map).view(bs, 2)

            # Z (Fused + Depth)
            z_input = torch.cat([fused_feat, depth_feat], dim=1)
            pred_z_map = self.z_head(z_input)
            pred_z = self.global_pool(pred_z_map).view(bs, 1)

            # 5. Combinazione della Traslazione Finale [X, Y, Z]
            pred_t = torch.cat([pred_xy, pred_z], dim=1) # [B, 3]

            return pred_r, pred_t