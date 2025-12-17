import torch
import torch.nn as nn
from torchvision import models
import numpy as np
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block per calibrare l'importanza dei canali."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PoseResNetRGBD(nn.Module):
    def __init__(self, pretrained=True, intrinsics=None):
        super(PoseResNetRGBD, self).__init__()
        """
        Modello PoseNet che utilizza input RGB-D (4 canali).
        - Input: Immagine RGB-D (4 canali)
        - Output: Quaternione (4) + Traslazione (3)
        intrinsics: Lista o tensor con i parametri intrinseci della camera [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        se None, usa valori di default LINEMOD.
        """
        
        # Carichiamo il backbone ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        if intrinsics is None:
            intrinsics = [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]
        self.fx = intrinsics[0]
        self.fy = intrinsics[4]
        self.cx = intrinsics[2]
        self.cy = intrinsics[5]

        self.register_buffer('cam_constants', torch.tensor([self.fx, self.fy, self.cx, self.cy], dtype=torch.float32))
        # --- MODIFICA 1: Input a 4 canali (RGB + Depth) ---
        # Prendiamo i pesi del primo strato (3 canali) e inizializziamo il 4° canale 
        # con la media dei primi tre o con una copia del canale red.
        existing_layer = resnet.conv1
        new_conv1 = nn.Conv2d(4, existing_layer.out_channels, 
                             kernel_size=existing_layer.kernel_size, 
                             stride=existing_layer.stride, 
                             padding=existing_layer.padding, 
                             bias=existing_layer.bias)
        
        with torch.no_grad():
            new_conv1.weight[:, :3, :, :] = existing_layer.weight
            # Inizializziamo il canale Depth con la media degli altri per non sbilanciare i gradienti all'inizio
            new_conv1.weight[:, 3, :, :] = existing_layer.weight.mean(dim=1)
        
        resnet.conv1 = new_conv1
        
        # --- MODIFICA 2: Inserimento SE Blocks ---
        # Inseriamo SE blocks dopo i layer principali per gestire l'interdipendenza RGB-D
        self.layer1 = resnet.layer1
        self.se1 = SEBlock(256)
        
        self.layer2 = resnet.layer2
        self.se2 = SEBlock(512)
        
        self.layer3 = resnet.layer3
        self.se3 = SEBlock(1024)
        
        self.layer4 = resnet.layer4
        self.se4 = SEBlock(2048)

        # Raggruppiamo la parte iniziale e finale
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.avgpool = resnet.avgpool
        feature_dim = 2048 
        # Input: 2 (bbox center dinamico) + 4 (costanti camera) = 6
        self.info_fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )
        
        combined_dim = feature_dim + 64
        
        self.rotation_head = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4)
        )
        
        self.translation_head = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3)
        )

    def forward(self, x, bbox_center):
        # x deve avere shape (B, 4, H, W)
        
        # Feature extraction con SE Blocks
        x = self.initial(x)
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        
        
        # 2. Smistamento e Concatenazione delle info geometriche
        batch_size = x.size(0)
        # Ripetiamo le costanti per ogni elemento del batch
        static_info = self.cam_constants.unsqueeze(0).expand(batch_size, -1)
        
        # Vettore completo [B, 6]: [center_x, center_y, fx, fy, cx, cy]
        geom_info = torch.cat([bbox_center, static_info], dim=1)
        geom_feat = self.info_fc(geom_info)
        
        # 3. Fusion e Predizione
        combined = torch.cat((x, geom_feat), dim=1)
        
        rot = self.rotation_head(combined)
        trans = self.translation_head(combined)
        
        # Normalizzazione quaternione
        rot = torch.nn.functional.normalize(rot, p=2, dim=1)
        
        return rot, trans