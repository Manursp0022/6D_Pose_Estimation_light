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
from models.PoseRefine_Net import PoseRefineNet
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset

class RefineTrainer:
    def __init__(self, config):
        self.cfg = config
        
        # Setup Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.num_workers = 8
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.num_workers = 2
        else:
            self.device = torch.device("cpu")
            self.num_workers = 0
            
        print(f"Initializing RefineTrainer on: {self.device}")
        
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        self.model_main = self._setup_main_model()
        self.refiner = self._setup_refiner()
        
        self.models_tensor = self._load_3d_models_tensor()
        
        self.train_loader = self._setup_data()

        self.optimizer = optim.Adam(self.refiner.parameters(), lr=self.cfg['lr'])
        self.criterion_t = nn.L1Loss() 

    def _setup_data(self):
        print("Loading Dataset...")
        train_ds = LineModPoseDataset(self.cfg['split_train'], self.cfg['dataset_root'], mode='train')
        return DataLoader(train_ds, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def quaternion_to_matrix(self, quaternions):
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
            ), -1)
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def _load_3d_models_tensor(self):
        print("Loading 3D Models into GPU Tensor...")
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

# --- LA RIVOLUZIONE: POINT MATCHING LOSS ---
    def criterion_add(self, pred_r, pred_t, gt_r, gt_t, model_points):
        """
        Calcola la distanza media tra i punti del modello trasformati con la posa predetta
        e quelli trasformati con la posa vera.
        Accoppia R e T in un'unica metrica fisica.
        """
        pred_R_mat = self.quaternion_to_matrix(pred_r) # [B, 3, 3]
        gt_R_mat = self.quaternion_to_matrix(gt_r)     # [B, 3, 3]
        
        # Trasforma nuvola PREDETTA: (R_pred * P) + t_pred
        # model_points: [B, N, 3] -> permute [B, 3, N]
        pred_pts = torch.bmm(pred_R_mat, model_points.permute(0, 2, 1)) + pred_t.unsqueeze(2) # [B, 3, N]
        
        # Trasforma nuvola VERA: (R_gt * P) + t_gt
        gt_pts = torch.bmm(gt_R_mat, model_points.permute(0, 2, 1)) + gt_t.unsqueeze(2)       # [B, 3, N]
        
        # Distanza Euclidea tra ogni punto corrispondente
        # distance: [B, 3, N]
        diff = pred_pts - gt_pts
        dis = torch.norm(diff, dim=1) # [B, N] (Norma L2 su xyz)
        
        # Loss Media su tutti i punti e su tutto il batch
        loss = torch.mean(dis)

        return loss

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
            class_ids = batch['class_id'].to(self.device)
            
            # A. Stima Iniziale
            with torch.no_grad():
                pred_r_quat, pred_t, emb_global = self.model_main.forward_refine(images, depth) 
            
            # B. Input Refiner
            model_points = self.models_tensor[class_ids.long()] # [B, 500, 3]
            
            R_pred = self.quaternion_to_matrix(pred_r_quat)
            rotated_pts = torch.bmm(R_pred, model_points.permute(0, 2, 1)) 
            cloud_input = rotated_pts + pred_t.unsqueeze(2) 
            
            # C. Refiner Forward
            delta_r, delta_t = self.refiner(cloud_input, emb_global)
            
            # D. Applicazione Delta
            refined_t = pred_t + delta_t
            refined_r = pred_r_quat + delta_r 
            refined_r = torch.nn.functional.normalize(refined_r, p=2, dim=1)
            
            # E. LOSS ADD (Point Matching)
            # Qui la magia: Se sposti la T male, i punti si allontanano -> Loss sale.
            # Se ruoti male la R, i punti si allontanano -> Loss sale.
            loss = self.criterion_add(refined_r, refined_t, gt_r_quat, gt_t, model_points)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            # Stampiamo la loss in cm (più leggibile)
            pbar.set_postfix({'ADD Loss (m)': f"{total_loss/steps:.4f}"})

        avg_loss = total_loss / steps if steps > 0 else 0
        return avg_loss

    def run(self):
        print("Starting Training Loop (ADD LOSS)...")
        best_loss = float('inf')
        
        for epoch in range(self.cfg['epochs']):
            epoch_loss = self.train_epoch(epoch)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                path = os.path.join(self.cfg['save_dir'], "best_refiner.pth")
                torch.save(self.refiner.state_dict(), path)
                print(f" >>> New Best Model Saved! ADD Loss: {best_loss:.5f}")
            
            if (epoch+1) % 5 == 0:
                path = os.path.join(self.cfg['save_dir'], f"refiner_ep{epoch+1}.pth")
                torch.save(self.refiner.state_dict(), path)