import torch
import torch.nn as nn
import utils.Posenet_utils.attention import GeometricAttention

class FeatureRefiner(nn.Module):
    """
    Refiner Iterativo 'Pure-Feature'.
    Invece di trasformare nuvole di punti, usa l'Attention per raffinare
    le feature basandosi sulla stima della posa corrente.
    """
    def __init__(self, in_channels=512):
        super().__init__()
        
        # 1. Pose Encoder: Trasforma la posa (7 valori) in un vettore feature
        self.pose_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels)
        )
        
        # 2. Fusion (Feature Immagine + Posa Corrente)
        # Input: Feature (512) + PoseEmb (512) = 1024
        self.fusion = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        
        # 3. Iterative Attention (Cuore del Paper)
        # La rete impara a "guardare" le parti dell'immagine che suggeriscono
        # un errore nella posa corrente.
        self.att = GeometricAttention(in_channels=in_channels)
        
        # 4. Delta Predictor
        self.fc_delta_r = nn.Linear(in_channels, 4)
        self.fc_delta_t = nn.Linear(in_channels, 3)
        
        # Init a zero per stabilità iniziale
        self.fc_delta_r.weight.data.zero_()
        self.fc_delta_r.bias.data.zero_()
        self.fc_delta_t.weight.data.zero_()
        self.fc_delta_t.bias.data.zero_()

    def forward(self, dense_feat, current_r, current_t):
        """
        dense_feat: [B, 512, N] Feature estratte dalla Baseline (Statiche)
        current_r, current_t: Posa corrente da raffinare
        """
        bs, c, n = dense_feat.size()
        
        # A. Codifica la Posa Corrente
        pose_vec = torch.cat([current_r, current_t], dim=1) # [B, 7]
        pose_emb = self.pose_encoder(pose_vec) # [B, 512]
        
        # Espandi per concatenare alle feature dense
        pose_emb_expanded = pose_emb.unsqueeze(2).repeat(1, 1, n) # [B, 512, N]
        
        # B. Condizionamento: "Data questa posa, cosa c'è di strano nell'immagine?"
        combined = torch.cat([dense_feat, pose_emb_expanded], dim=1) # [B, 1024, N]
        x = self.fusion(combined) # [B, 512, N]
        
        # C. Attention Iterativa
        # Applica l'attenzione per pesare i punti più rilevanti per l'errore
        att_map = self.att(x.unsqueeze(-1)).squeeze(-1) # [B, 1, N]
        x_weighted = x * (1 + att_map)
        
        # D. Global Pooling
        # Usiamo Max Pooling per trovare la feature "più in disaccordo"
        global_feat = torch.max(x_weighted, dim=2)[0] # [B, 512]
        
        # E. Predizione Delta
        delta_r = self.fc_delta_r(global_feat)
        delta_t = self.fc_delta_t(global_feat)
        
        return delta_r, delta_t