import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseRefineNet(nn.Module):
    def __init__(self, num_points=1000, num_obj=13):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        
        # Input: Feature Globali (1024) + Punti 3D trasformati (3) = 1027 canali
        # Nel DenseFusion originale concatenano emb_global e i punti
        self.conv1 = nn.Conv1d(512 + 3, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        
        # Output: Correzione Delta R (4) e Delta T (3)
        self.conv_r = nn.Conv1d(128, 4, 1)
        self.conv_t = nn.Conv1d(128, 3, 1)

    def forward(self, x, emb_global):
        # x: Punti nuvola trasformati con la posa corrente [BS, 3, N]
        # emb_global: Feature globali dalla rete DenseFusion [BS, 1024, 1]
        
        bs, _, num_points = x.size()
        
        # Replica l'embedding globale per ogni punto
        emb_global = emb_global.unsqueeze(2) # [BS, 1024, N]
        emb_global_expanded = emb_global.repeat(1, 1, num_points)
        
        # Concatena: "Cosa vedo (feature)" + "Dove penso di essere (punti)"
        inp = torch.cat((x, emb_global_expanded), dim=1) # [BS, 1027, N]
        
        out = F.relu(self.conv1(inp))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out)) # [BS, 128, N]
        
        # Max Pooling per avere un vettore globale di correzione
        out = torch.max(out, 2, keepdim=True)[0] # [BS, 128, 1]
        
        # Predici Delta
        r = self.conv_r(out).view(bs, 4)
        t = self.conv_t(out).view(bs, 3)
        
        # Il quaternione deve essere unitario
        r = F.normalize(r, p=2, dim=1)
        
        return r, t