import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, channels=512, reduction=16, prior_strength=0.4):
        super().__init__()
        
        self.prior_strength = prior_strength  # Quanto pesa il prior (0 = niente, 1 = solo prior)
        
        # Creiamo il prior gaussiano centrato (verrà registrato come buffer, non come parametro)
        self.register_buffer('gaussian_prior', self._create_gaussian_prior(7, 7))
        
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels // reduction, channels // reduction, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, kernel_size=1),
            # Nota: Rimuoviamo Sigmoid qui, la applichiamo dopo aver combinato col prior
        )
        
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, kernel_size=1),
            nn.Sigmoid()
        )

    def _create_gaussian_prior(self, h, w, sigma=1.5):
        """Crea una gaussiana 2D centrata normalizzata tra 0 e 1"""
        y = torch.arange(h).float() - (h - 1) / 2
        x = torch.arange(w).float() - (w - 1) / 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.max()  # Normalizza a [0, 1]
        
        return gaussian.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def forward(self, rgb_feat, depth_feat):
        B = rgb_feat.size(0)
        
        combined = torch.cat([rgb_feat, depth_feat], dim=1)
        
        # --- A. SPATIAL ATTENTION con PRIOR ---
        x_compressed = self.channel_reducer(combined)
        learned_attention = self.spatial_gate(x_compressed)  # [B, 1, 7, 7] (logits, no sigmoid yet)
        
        # Combiniamo: attention appresa + prior gaussiano
        # Il prior viene espanso per il batch
        prior = self.gaussian_prior.expand(B, -1, -1, -1)  # [B, 1, 7, 7]
        
        # Metodo semplice: media pesata prima della sigmoid
        combined_attention = learned_attention + self.prior_strength * (prior * 2 - 1)  # prior scalato a [-1, 1]
        spatial_mask = torch.sigmoid(combined_attention)  # [B, 1, 7, 7]
        
        # --- B. CHANNEL ATTENTION ---
        channel_scale = self.channel_gate(combined)
        rgb_scale, depth_scale = torch.split(channel_scale, rgb_feat.size(1), dim=1)
        
        # --- C. APPLICAZIONE ---
        rgb_attended = rgb_feat * spatial_mask
        depth_attended = depth_feat * spatial_mask
        
        rgb_final = rgb_attended * rgb_scale
        depth_final = depth_attended * depth_scale
        
        return rgb_final, depth_final, spatial_mask