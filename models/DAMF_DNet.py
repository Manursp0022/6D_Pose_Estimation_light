import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from utils.Posenet_utils.attention import GeometricAttention 

class DAMF_Net(nn.Module):
    def __init__(self, pretrained=True, temperature=1.0):
        super().__init__()

        self.temperature = temperature
        self.eps = 1e-8

        # --- 1. ResNet34 SENZA DILATION (per evitare errori) ---
        # Output standard: 7x7 con 512 channels
        self.rgb_backbone = models.resnet18(
            weights='DEFAULT' if pretrained else None
        )
        self.depth_backbone = models.resnet18(
            weights='DEFAULT' if pretrained else None
        )

        # Estrai layer intermedi per multi-scale features
        self.rgb_layer3 = nn.Sequential(
            *list(self.rgb_backbone.children())[:7]  # fino a layer3: 28x28, 256ch
        )
        self.rgb_layer4 = self.rgb_backbone.layer4  # 7x7, 512ch
        
        self.depth_layer3 = nn.Sequential(
            *list(self.depth_backbone.children())[:7]
        )
        self.depth_layer4 = self.depth_backbone.layer4
        
        # --- 2. LIGHTWEIGHT DECODER per recuperare risoluzione ---
        # Porta da 7x7 (512ch) a 14x14 (256ch)
        self.decoder = nn.Sequential(
            # Upsample 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # --- 3. ATTENTION (su feature 256ch dopo decoder) ---
        self.attention_block = GeometricAttention(in_channels=256)
        
        # --- 4. FUSION ---
        # Skip connection: concatena layer3 (256) con decoder output (256)
        # Input: 256 (layer3_rgb) + 256 (decoded_rgb) + 256 (layer3_depth) + 256 (decoded_depth) = 1024
        self.fusion_entry = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        self.fusion_res = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.BatchNorm2d(512)
        )
        
        # --- HEADS DECOUPLED ---
        # Rotazione: fused (512) + rgb_features (512) = 1024
        self.rot_head = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1), 
            nn.ReLU(),
            nn.Conv2d(128, 4, 1)    
        )
        
        # Traslazione: fused (512) + depth_features (512) = 1024
        self.trans_head = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1), 
            nn.ReLU(),
            nn.Conv2d(128, 3, 1)    
        )
        
        # Confidence: usa solo fused (512)
        self.conf_head = nn.Sequential(
            nn.Conv2d(512, 128, 1), 
            nn.ReLU(),
            nn.Conv2d(128, 1, 1) 
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")

    def _forward_fusion(self, rgb, depth, mask=None):
        """
        Estrae feature multi-scale con decoder per recuperare risoluzione.
        
        Returns:
            fused_feat: [B, 512, 14, 14]
            rgb_features: [B, 512, 14, 14] - per skip connection a rot_head
            depth_features: [B, 512, 14, 14] - per skip connection a trans_head
        """
        if mask is not None:
            rgb = rgb * mask
            depth = depth * mask
        
        # --- Extract multi-scale features ---
        # Layer3: 28x28, 256 channels (dettagli intermedi)
        rgb_l3 = self.rgb_layer3(rgb)  # [B, 256, 28, 28]
        depth_3ch = torch.cat([depth, depth, depth], dim=1)
        depth_l3 = self.depth_layer3(depth_3ch)  # [B, 256, 28, 28]
        
        # Layer4: 7x7, 512 channels (semantica alta)
        rgb_l4 = self.rgb_layer4(rgb_l3)  # [B, 512, 7, 7]
        depth_l4 = self.depth_layer4(depth_l3)  # [B, 512, 7, 7]
        
        # --- Decoder: 7x7 -> 14x14 ---
        rgb_decoded = self.decoder(rgb_l4)  # [B, 256, 14, 14]
        depth_decoded = self.decoder(depth_l4)  # [B, 256, 14, 14]
        
        # --- Downsample layer3 da 28x28 a 14x14 per matching ---
        rgb_l3_down = F.adaptive_avg_pool2d(rgb_l3, (14, 14))  # [B, 256, 14, 14]
        depth_l3_down = F.adaptive_avg_pool2d(depth_l3, (14, 14))  # [B, 256, 14, 14]
        
        # --- Geometric Attention (su depth decoded) ---
        att_map = self.attention_block(depth_decoded)  # [B, 1, 14, 14]
        
        # Combina layer3 + decoded con attention
        rgb_combined = torch.cat([rgb_l3_down, rgb_decoded], dim=1)  # [B, 512, 14, 14]
        depth_combined = torch.cat([depth_l3_down, depth_decoded], dim=1)  # [B, 512, 14, 14]
        
        rgb_enhanced = rgb_combined * (1 + att_map) 
        depth_enhanced = depth_combined * (1 + att_map) 
        
        # --- Fusion ---
        combined = torch.cat([rgb_enhanced, depth_enhanced], dim=1)  # [B, 1024, 14, 14]
        x = self.fusion_entry(combined)  # [B, 512, 14, 14]
        x_res = self.fusion_res(x)
        fused_feat = F.relu(x + x_res)  # [B, 512, 14, 14]
        
        return fused_feat, rgb_enhanced, depth_enhanced

    def _weighted_pooling(self, fused_feat, rgb_enhanced, depth_enhanced):
        """
        Weighted pooling con confidence per ottenere pose globale.
        """
        batch_size = fused_feat.size(0)
        
        # --- DECOUPLED INPUTS con Skip Connections ---
        rot_input = torch.cat([fused_feat, rgb_enhanced], dim=1)  # [B, 1024, 14, 14]
        pred_rot_map = self.rot_head(rot_input)  # [B, 4, 14, 14]
        
        trans_input = torch.cat([fused_feat, depth_enhanced], dim=1)  # [B, 1024, 14, 14]
        pred_trans_map = self.trans_head(trans_input)  # [B, 3, 14, 14]
        
        conf_logits = self.conf_head(fused_feat)  # [B, 1, 14, 14]
        
        # --- POOLING con CONFIDENCE WEIGHTING ---
        pred_rot_map = pred_rot_map.view(batch_size, 4, -1)   # [B, 4, 196]
        pred_trans_map = pred_trans_map.view(batch_size, 3, -1)  # [B, 3, 196]
        conf_logits = conf_logits.view(batch_size, 1, -1)  # [B, 1, 196]
        
        # Normalizza quaternioni per pixel
        pred_rot_map = F.normalize(pred_rot_map + self.eps, p=2, dim=1)
        
        # Confidence weighting
        weights = F.softmax(conf_logits / self.temperature, dim=2)  # [B, 1, 196]
        
        # Weighted average
        pred_rot_global = torch.sum(pred_rot_map * weights, dim=2)  # [B, 4]
        pred_trans_global = torch.sum(pred_trans_map * weights, dim=2)  # [B, 3]
        
        # Normalizza quaternione finale
        pred_rot_global = F.normalize(pred_rot_global + self.eps, p=2, dim=1)
        
        return pred_rot_global, pred_trans_global

    def forward(self, rgb, depth, mask=None):
        """
        Forward pass completo.
        
        Args:
            rgb: [B, 3, H, W] - immagine RGB
            depth: [B, 1, H, W] - mappa di profondità
            mask: [B, 1, H, W] - maschera opzionale dell'oggetto
            
        Returns:
            pred_rot: [B, 4] - quaternione di rotazione
            pred_trans: [B, 3] - traslazione
        """
        fused_feat, rgb_enhanced, depth_enhanced = self._forward_fusion(rgb, depth, mask)
        pred_rot, pred_trans = self._weighted_pooling(fused_feat, rgb_enhanced, depth_enhanced)
        
        return pred_rot, pred_trans
    
    def get_confidence_map(self, rgb, depth, mask=None):
        """
        Ritorna la mappa di confidence spaziale (utile per visualizzazione).
        
        Returns:
            conf_map: [B, 1, 14, 14] - confidence tra 0 e 1
        """
        fused_feat, _, _ = self._forward_fusion(rgb, depth, mask)
        conf_logits = self.conf_head(fused_feat)
        conf_map = torch.sigmoid(conf_logits)
        return conf_map
    
    def forward_with_debug(self, rgb, depth, mask=None):
        """
        Forward con informazioni di debug per analisi.
        """
        fused_feat, rgb_enhanced, depth_enhanced = self._forward_fusion(rgb, depth, mask)
        pred_rot, pred_trans = self._weighted_pooling(fused_feat, rgb_enhanced, depth_enhanced)
        
        # Estrai attention map
        depth_3ch = torch.cat([depth, depth, depth], dim=1)
        depth_l3 = self.depth_layer3(depth_3ch)
        depth_l4 = self.depth_layer4(depth_l3)
        depth_decoded = self.decoder(depth_l4)
        att_map = self.attention_block(depth_decoded)
        
        debug_info = {
            'fused_features': fused_feat,
            'rgb_enhanced': rgb_enhanced,
            'depth_enhanced': depth_enhanced,
            'attention_map': att_map,
            'confidence_map': self.get_confidence_map(rgb, depth, mask)
        }
        
        return pred_rot, pred_trans, debug_info