import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import numpy as np
from plyfile import PlyData

# Import dei Modelli
from models.DenseFusion_RGBD_Net import DenseFusion_RGBD_Net
from models.RefineNet import PoseRefineNet
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset

class RefineTrainer:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing RefineTrainer on: {self.device}")
        
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        # 1. SETUP MODELLI
        self.model_main = self._setup_main_model()
        self.refiner = self._setup_refiner()
        
        # 2. CARICA GEOMETRIA (Vettorizzata)
        # Creiamo un tensore unico [Max_ID+1, 500, 3] per accesso veloce GPU
        self.models_tensor = self._load_3d_models_tensor()
        
        # 3. SETUP DATI
        self.train_loader = self._setup_data()
        
        # 4. OPTIMIZER & LOSS
        self.optimizer = optim.Adam(self.refiner.parameters(), lr=self.cfg['lr'])
        self.criterion_L1 = nn.L1Loss()

    def _setup_data(self):
        print("Loading Dataset...")
        train_ds = LineModPoseDataset(self.cfg['split_train'], self.cfg['dataset_root'], mode='train')
        
        # --- MODIFICA A100: num_workers=8 ---
        # Usa 8 o 16 su un server potente. 
        return DataLoader(train_ds, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=8, pin_memory=True)

    def quaternion_to_matrix(self, quaternions):
        """Converte quaternioni [B, 4] in matrici rotazione [B, 3, 3]"""
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def _load_3d_models_tensor(self):
        print("Loading 3D Models into GPU Tensor...")
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        # Linemod ha ID fino a 15. Creiamo un buffer fino a 16.
        # Shape: [16, 500, 3]
        max_id = 16 
        num_pts = self.cfg['num_points_mesh']
        
        # Inizializziamo a zero
        all_models = torch.zeros((max_id, num_pts, 3), dtype=torch.float32)
        
        obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        for obj_id in obj_ids:
            path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
            if os.path.exists(path):
                ply = PlyData.read(path)
                v = ply['vertex']
                pts = np.stack([v['x'], v['y'], v['z']], axis=-1) / 1000.0 # Metri
                
                # Sottocampionamento
                if pts.shape[0] > num_pts:
                    idx = np.random.choice(pts.shape[0], num_pts, replace=False)
                    pts = pts[idx, :]
                elif pts.shape[0] < num_pts:
                    # Pad con duplicati se ne ha pochi (raro)
                    diff = num_pts - pts.shape[0]
                    idx = np.random.choice(pts.shape[0], diff, replace=True)
                    pts = np.concatenate([pts, pts[idx]], axis=0)
                
                all_models[obj_id] = torch.from_numpy(pts).float()
            else:
                print(f"Warning: Model {obj_id} not found at {path}")
                
        return all_models.to(self.device)

    def _setup_main_model(self):
        print("Loading Frozen Main Model...")
        model = DenseFusion_RGBD_Net(pretrained=False).to(self.device)
        model.load_state_dict(torch.load(self.cfg['main_weights'], map_location=self.device))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _setup_refiner(self):
        print("Initializing RefineNet...")
        refiner = PoseRefineNet(num_points=self.cfg['num_points_mesh']).to(self.device)
        return refiner

    def train_epoch(self, epoch_idx):
        self.refiner.train()
        total_loss = 0
        steps = 0
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch_idx+1}")
        
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            depth = batch['depth'].to(self.device, non_blocking=True)
            gt_t = batch['translation'].to(self.device, non_blocking=True)
            gt_r_quat = batch['quaternion'].to(self.device, non_blocking=True)
            class_ids = batch['class_id'].to(self.device) # Ora è un tensore GPU
            
            bs = images.size(0)

            # A. Stima Iniziale (Frozen)
            with torch.no_grad():
                pred_r_quat, pred_t, emb_global = self.model_main.forward_refine(images, depth) 
            
            # B. Preparazione Input Geometrico (VETTORIZZATA ⚡️)
            # 1. Recupera i punti modello per tutto il batch in un colpo solo
            # class_ids serve da indice: [B] -> [B, 500, 3]
            # Assicurati che class_ids sia LongTensor
            model_points = self.models_tensor[class_ids.long()] # [B, 500, 3]
            
            # 2. Converti Quaternioni Predetti in Matrici
            R_pred = self.quaternion_to_matrix(pred_r_quat) # [B, 3, 3]
            
            # 3. Applica Trasformazione: (R @ Punti.T).T + t
            # Punti: [B, 500, 3] -> Permute [B, 3, 500] per moltiplicare con R [B, 3, 3]
            # Risultato: [B, 3, 500]
            rotated_pts = torch.bmm(R_pred, model_points.permute(0, 2, 1)) 
            
            # Somma traslazione (broadcasting su ultima dimensione)
            # t: [B, 3] -> [B, 3, 1]
            cloud_input = rotated_pts + pred_t.unsqueeze(2) # [B, 3, 500]
            
            # C. Refiner Forward
            # Refiner si aspetta [B, 3, N], che è esattamente 'cloud_input'
            delta_r, delta_t = self.refiner(cloud_input, emb_global)
            
            # D. Loss
            refined_t = pred_t + delta_t
            refined_r = pred_r_quat + delta_r 
            refined_r = torch.nn.functional.normalize(refined_r, p=2, dim=1)
            
            loss_t = self.criterion_L1(refined_t, gt_t)
            loss_r = self.criterion_L1(refined_r, gt_r_quat)
            
            loss = loss_t + loss_r
            
            # E. Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({'Loss': total_loss/steps})

        avg_loss = total_loss / steps if steps > 0 else 0
        return avg_loss

    def run(self):
        print("Starting Training Loop (A100 Optimized)...")
        best_loss = float('inf')
        
        for epoch in range(self.cfg['epochs']):
            epoch_loss = self.train_epoch(epoch)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                path = os.path.join(self.cfg['save_dir'], "best_refiner.pth")
                torch.save(self.refiner.state_dict(), path)
                print(f" >>> New Best Model Saved! Loss: {best_loss:.5f}")
            
            if (epoch+1) % 5 == 0:
                path = os.path.join(self.cfg['save_dir'], f"refiner_ep{epoch+1}.pth")
                torch.save(self.refiner.state_dict(), path)