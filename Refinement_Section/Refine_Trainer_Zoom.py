import torch
import torch.optim as optim
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from plyfile import PlyData
import numpy as np

# TUOI IMPORT
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net 
from utils.Posenet_utils.DenseFusion_Loss import DenseFusionLoss

class RefineTrainerZoom:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. MODELLO (Stessa architettura della Baseline)
        print("Initializing Refiner Network...")
        self.model = DenseFusion_Masked_DualAtt_Net(pretrained=True, temperature=2.0).to(self.device)
        
        # Se vuoi partire dai pesi della baseline (Consigliato: converge prima)
        if 'main_weights' in self.cfg:
            print(f"Loading weights from {self.cfg['main_weights']}")
            state = torch.load(self.cfg['main_weights'], map_location=self.device)
            # Pulisci le chiavi _orig_mod se necessario
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
            self.model.load_state_dict(state)

        # 2. DATASET CON RUMORE ALTO (Simula l'errore di 4cm)
        print("Loading High-Noise Dataset for Refinement...")
        # noise_factor=0.20 significa +/- 20% di errore sul crop.
        # Su un oggetto di 100px, sono 20px di spostamento -> copre abbondantemente i 4cm.
        self.train_ds = LineModPoseDataset(
            self.cfg['split_train'], self.cfg['dataset_root'], mode='train', 
            noise_factor=0.20  
        )
        self.train_loader = DataLoader(self.train_ds, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        
        # 3. LOSS & OPTIMIZER
        self.criterion = DenseFusionLoss(self.device)
        self.models_tensor = self._load_3d_models_tensor()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['lr'])
        self.scaler = torch.cuda.amp.GradScaler()

    def _load_3d_models_tensor(self):
            # (Codice identico a prima per caricamento ply)
            print("Loading 3D Models into VRAM...")
            models_dir = os.path.join(self.cfg['dataset_root'], 'models')
            max_id = 16 
            num_pts = self.cfg['num_points_mesh']
            all_models = torch.zeros((max_id, num_pts, 3), dtype=torch.float32)
            obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
            for obj_id in obj_ids:
                path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
                if os.path.exists(path):
                    ply = PlyData.read(path)
                    v = ply['vertex']
                    pts = np.stack([v['x'], v['y'], v['z']], axis=-1) / 1000.0
                    if pts.shape[0] > num_pts:
                        idx = np.random.choice(pts.shape[0], num_pts, replace=False)
                        pts = pts[idx, :]
                    all_models[obj_id] = torch.from_numpy(pts).float()
            return all_models.to(self.device)

    def train(self):
        print(f"Starting Refiner Training ({self.cfg['epochs']} epochs)...")
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        for epoch in range(self.cfg['epochs']):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Refine Ep {epoch+1}")
            ep_loss = 0
            steps = 0
            
            for batch in loop:
                img = batch['image'].to(self.device)
                depth = batch['depth'].to(self.device)
                mask = batch['mask'].to(self.device)
                gt_t = batch['translation'].to(self.device)
                gt_q = batch['quaternion'].to(self.device)
                cls_ids = batch['class_id'].to(self.device)
                
                self.optimizer.zero_grad()
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # La rete riceve un crop "sbagliato" (fatto dal dataset) e deve predire la GT
                    pred_r, pred_t = self.model(img, depth, mask)
                    pts = self.models_tensor[cls_ids.long()]
                    loss = self.criterion(pred_r, pred_t, gt_q, gt_t, pts, cls_ids)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                ep_loss += loss.item()
                steps += 1
                loop.set_postfix(loss=loss.item())
            
            avg_loss = ep_loss / steps
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
            
            # Salva
            torch.save(self.model.state_dict(), os.path.join(self.cfg['save_dir'], "best_refiner_net.pth"))

if __name__ == "__main__":
    config = {
        'dataset_root': "/content/dataset/Linemod_preprocessed",
        'split_train': "/content/6D_Pose_Estimation_light/data/autosplit_train_ALL.txt",
        'save_dir': "checkpoints_refine/",
        'main_weights': "checkpoints/best_turbo_model_A100.pth", # Parti dalla baseline!
        'batch_size': 64, 
        'lr': 0.0001,
        'epochs': 20 # Bastano poche epoche
    }
    trainer = RefineTrainerZoom(config)
    trainer.train()