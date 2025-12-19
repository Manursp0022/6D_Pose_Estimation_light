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
    def __init__(self,config):
        # --- CONFIGURAZIONE ---
        self.cfg = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing RefineTrainer on: {self.device}")
        
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        # 1. SETUP DATI
        self.train_loader = self._setup_data()
        self.models_3d = self._load_3d_models()
        
        # 2. SETUP MODELLI
        self.model_main = self._setup_main_model()
        self.refiner = self._setup_refiner()
        
        # 3. OPTIMIZER & LOSS
        self.optimizer = optim.Adam(self.refiner.parameters(), lr=self.cfg['lr'])
        self.criterion_L1 = nn.L1Loss() # Per semplicità usiamo L1 su delta
    

    def _setup_data(self):
        print("Loading Dataset...")
        train_ds = LineModPoseDataset(self.cfg['split_train'], self.cfg['dataset_root'], mode='train')
        # num_workers=0 per Mac
        return DataLoader(train_ds, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=0)
    
    def quaternion_to_matrix(self, quaternions):
            """
            Versione PyTorch nativa: Mantiene i gradienti e lavora su GPU.
            """
            # Scompattiamo w, x, y, z (o x, y, z, w a seconda dell'ordine)
            # Assumiamo ordine standard [x, y, z, w] se usi scipy come riferimento, 
            # MA PyTorch DenseFusion di solito esce [w, x, y, z] o [x, y, z, w].
            # Controlliamo l'ordine: Solitamente DenseFusion outputta quaternioni normalizzati.
            
            # Qui assumiamo output [w, x, y, z] (Parte reale prima). 
            # Se i tuoi risultati sono strani, prova a cambiare l'ordine in r, i, j, k = ...
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

    def _load_3d_models(self):
        print("Loading 3D Models (PLY) for Geometry...")
        models_3d = {}
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        # Mapping ID (i tuoi ID dataset)
        obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        
        for obj_id in obj_ids:
            path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
            if os.path.exists(path):
                ply = PlyData.read(path)
                v = ply['vertex']
                pts = np.stack([v['x'], v['y'], v['z']], axis=-1) / 1000.0 # Metri
                
                # Sottocampionamento casuale (per velocità)
                if pts.shape[0] > self.cfg['num_points_mesh']:
                    idx = np.random.choice(pts.shape[0], self.cfg['num_points_mesh'], replace=False)
                    pts = pts[idx, :]
                
                models_3d[obj_id] = torch.from_numpy(pts).float().to(self.device)
        return models_3d

    def _setup_main_model(self):
        print("Loading Frozen Main Model...")
        model = DenseFusion_RGBD_Net(pretrained=False).to(self.device)
        model.load_state_dict(torch.load(self.cfg['main_weights'], map_location=self.device))
        model.eval()
        # CONGELA TUTTO
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _setup_refiner(self):
        print("Initializing RefineNet...")
        # Nota: Input channels = 512 (feature) + 3 (xyz)
        refiner = PoseRefineNet(num_points=self.cfg['num_points_mesh']).to(self.device)
        return refiner

    def train_epoch(self, epoch_idx):
        self.refiner.train()
        total_loss = 0
        steps = 0
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch_idx+1}")
        for batch in pbar:
            images = batch['image'].to(self.device)
            depth = batch['depth'].to(self.device)
            gt_t = batch['translation'].to(self.device) # [B, 3]
            gt_r_quat = batch['quaternion'].to(self.device) # [B, 4]
            class_ids = batch['class_id']
            
            bs = images.size(0)

            # A. Stima Iniziale (Frozen)
            with torch.no_grad():
                # Qui usiamo il nuovo metodo forward_refine!
                pred_r_quat, pred_t, emb_global = self.model_main.forward_refine(images, depth) 
                # emb_global shape: [B, 512]
            
            # B. Preparazione Input Geometrico per Refiner
            # Dobbiamo creare la nuvola di punti trasformata con la posa PREDETTA
            cloud_input_list = []
            
            for k in range(bs):
                oid = int(class_ids[k])
                if oid not in self.models_3d:
                    # Fallback (non dovrebbe succedere)
                    dummy = torch.zeros((self.cfg['num_points_mesh'], 3), device=self.device)
                    cloud_input_list.append(dummy)
                    continue
                
                # Prendi punti modello base
                raw_pts = self.models_3d[oid] # [N, 3]
                
                # Trasforma con Predizione Iniziale (che contiene errore)
                # Conversione Quat -> Matrix (Semplificata per il loop)
                # (Idealmente usa una funzione batch, qui facciamo loop per chiarezza)
                R_pred = self.quaternion_to_matrix(pred_r_quat[k]) # [3, 3]
                t_pred = pred_t[k] # [3]
                
                # Applica: R*x + t
                transformed_pts = torch.mm(raw_pts, R_pred.T) + t_pred # [N, 3]
                cloud_input_list.append(transformed_pts)
            
            # Stack -> [B, N, 3] -> Permute [B, 3, N] per Conv1d
            cloud_input = torch.stack(cloud_input_list).permute(0, 2, 1) # [B, 3, N]
            
            # C. Refiner Forward
            # Input: Punti trasformati "male" + Embedding visivo
            # Output: Delta R, Delta T
            delta_r, delta_t = self.refiner(cloud_input, emb_global)
            
            # D. Calcolo Loss (Semplificata: Supervised su quanto deve correggere)
            # La "vera" correzione necessaria è (GT - Pred)
            # Qui approssimiamo allenando la posa finale raffinata vs GT
            
            # Applica Delta alla predizione (Approssimazione lineare per training stabile)
            refined_t = pred_t + delta_t
            refined_r = pred_r_quat + delta_r # Somma brutale quaternioni (funziona per piccoli delta)
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

        print(f"Epoch {epoch_idx+1} finished. Avg Loss: {total_loss/steps:.4f}")

    def run(self):
        print("Starting Training Loop...")
        for epoch in range(self.cfg['epochs']):
            self.train_epoch(epoch)
            
            # Salva checkpoint ogni 5 epoche
            if (epoch+1) % 5 == 0:
                path = os.path.join(self.cfg['save_dir'], f"refiner_ep{epoch+1}.pth")
                torch.save(self.refiner.state_dict(), path)
                print(f"Saved checkpoint: {path}")