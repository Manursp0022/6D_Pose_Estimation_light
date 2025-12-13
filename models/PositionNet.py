import torch
import torch.nn as nn

class PositionNet(nn.Module):
    def __init__(self, num_classes=13, embedding_dim=16):
        super(PositionNet, self).__init__()
        
        # Embedding layer to convert class IDs to dense vectors
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        
        # input_dim = 9 (intrinsics) + 4 (box) + embedding_dim
        input_dim = 9 + 4 + embedding_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 3)  # Output: x, y, z
        )

    def forward(self, geom_data, class_ids):
        # geom_data: [Batch, 13] -> (9 intrinsics + 4 bbox)
        # class_ids: [Batch] -> indici interi delle classi
        
        # Otteniamo il vettore denso per la classe
        class_feat = self.class_embedding(class_ids) # [Batch, embedding_dim]
        
        # Uniamo i dati geometrici con l'embedding
        x = torch.cat((geom_data, class_feat), dim=1)
        
        return self.network(x)