import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionBBoxGuided(nn.Module):
    """
    Cross-Modal Attention con guida spaziale dalla Bounding Box.
    Elimina la necessità di maschere esterne usando la bbox come prior.
    """
    def __init__(self, channels=512, reduction=16, feature_size=7):
        super().__init__()
        
        self.feature_size = feature_size
        
        # 1. Channel Reducer per combinare RGB + Depth
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Spatial Attention Generator (impara a raffinare il prior della bbox)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels // reduction + 1, channels // reduction, kernel_size=3, padding=1),  # +1 per bbox prior
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 3. Channel Attention (quali filtri sono importanti)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 4. Learnable refinement del bbox prior
        self.prior_weight = nn.Parameter(torch.tensor(0.5))  # Quanto pesare il prior vs learned attention
        
        # Pre-compute coordinate grid (verrà spostato su device al primo forward)
        self.register_buffer('coord_x', None)
        self.register_buffer('coord_y', None)

    def _create_bbox_prior(self, bb_info, batch_size, device):
        """
        Crea una soft mask gaussiana dalla bounding box.
        
        Args:
            bb_info: [B, 4] tensor con [cx_norm, cy_norm, w_norm, h_norm]
            
        Returns:
            prior: [B, 1, H, W] soft attention prior
        """
        H, W = self.feature_size, self.feature_size
        
        # Crea coordinate grid se non esistono o sono su device sbagliato
        if self.coord_x is None or self.coord_x.device != device:
            y_coords = torch.linspace(0, 1, H, device=device)
            x_coords = torch.linspace(0, 1, W, device=device)
            self.coord_y, self.coord_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Estrai parametri bbox [B, 4] -> cx, cy, w, h
        cx = bb_info[:, 0].view(-1, 1, 1)  # [B, 1, 1]
        cy = bb_info[:, 1].view(-1, 1, 1)
        w = bb_info[:, 2].view(-1, 1, 1).clamp(min=0.05)   # Clamp per evitare divisione per zero
        h = bb_info[:, 3].view(-1, 1, 1).clamp(min=0.05)
        
        # Expand coordinate grid per batch
        coord_x = self.coord_x.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]
        coord_y = self.coord_y.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Calcola distanza normalizzata dal centro della bbox
        # Usiamo una gaussiana 2D ellittica
        dist_x = ((coord_x - cx) / (w / 2)) ** 2
        dist_y = ((coord_y - cy) / (h / 2)) ** 2
        
        # Gaussiana: exp(-0.5 * dist^2), ma usiamo sigma=1 per semplicità
        # Il risultato è ~1 al centro della bbox, ~0 ai bordi
        gaussian_prior = torch.exp(-0.5 * (dist_x + dist_y))  # [B, H, W]
        
        # Normalizza tra 0 e 1
        prior_min = gaussian_prior.view(batch_size, -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        prior_max = gaussian_prior.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        gaussian_prior = (gaussian_prior - prior_min) / (prior_max - prior_min + 1e-8)
        
        return gaussian_prior.unsqueeze(1)  # [B, 1, H, W]

    def forward(self, rgb_feat, depth_feat, bb_info):
        """
        Args:
            rgb_feat: [B, C, H, W] RGB features
            depth_feat: [B, C, H, W] Depth features
            bb_info: [B, 4] Bounding box [cx_norm, cy_norm, w_norm, h_norm]
            
        Returns:
            rgb_final: RGB features attese
            depth_final: Depth features attese
            spatial_mask: La maschera di attenzione finale (per debug/visualizzazione)
        """
        batch_size = rgb_feat.size(0)
        device = rgb_feat.device
        
        # 1. Crea bbox prior
        bbox_prior = self._create_bbox_prior(bb_info, batch_size, device)  # [B, 1, H, W]
        
        # 2. Concatena features
        combined = torch.cat([rgb_feat, depth_feat], dim=1)  # [B, 1024, H, W]
        
        # 3. Genera learned spatial attention (condizionata sul bbox prior)
        x_compressed = self.channel_reducer(combined)  # [B, 32, H, W]
        x_with_prior = torch.cat([x_compressed, bbox_prior], dim=1)  # [B, 33, H, W]
        learned_attention = self.spatial_gate(x_with_prior)  # [B, 1, H, W]
        
        # 4. Combina bbox prior con learned attention
        # prior_weight è learnable: la rete impara quanto fidarsi del prior
        alpha = torch.sigmoid(self.prior_weight)
        spatial_mask = alpha * bbox_prior + (1 - alpha) * learned_attention
        
        # 5. Channel attention
        channel_scale = self.channel_gate(combined)  # [B, 1024, 1, 1]
        rgb_scale, depth_scale = torch.split(channel_scale, rgb_feat.size(1), dim=1)
        
        # 6. Applica attention
        rgb_attended = rgb_feat * spatial_mask
        depth_attended = depth_feat * spatial_mask
        
        rgb_final = rgb_attended * rgb_scale
        depth_final = depth_attended * depth_scale
        
        return rgb_final, depth_final, spatial_mask