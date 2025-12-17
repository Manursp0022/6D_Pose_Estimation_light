import torch
import torch.nn as nn
from torchvision import models

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
    def __init__(self, pretrained=True):
        super(PoseResNetRGBD, self).__init__()
        
        # Carichiamo il backbone ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
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
        
        # --- MODIFICA 3: Due Teste di Regressione ---
        # Testa per la Rotazione (Quaternione)
        self.rotation_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 4)
        )
        
        # Testa per la Traslazione (X, Y, Z)
        self.translation_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 3)
        )

    def forward(self, x):
        # x deve avere shape (B, 4, H, W)
        
        # Feature extraction con SE Blocks
        x = self.initial(x)
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Head prediction
        rot = self.rotation_head(x)
        trans = self.translation_head(x)
        
        # Normalizzazione quaternione
        rot = torch.nn.functional.normalize(rot, p=2, dim=1)
        
        return rot, trans