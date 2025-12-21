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
        
        # --- 1. OTTIMIZZAZIONE HARDWARE A100 ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True # Ottimizza kernel conv
            # Ottimizzazione precisione matrici per A100 (Tensor Cores)
            torch.set_float32_matmul_precision('high') 
            print(f">>> A100 DETECTED. Using {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print(">>> WARNING: Using CPU. Training will be slow.")
        
        # --- 2. MODELLO ---
        print("Initializing Refiner Network...")
        self.model = DenseFusion_Masked_DualAtt_Net(pretrained=True, temperature=2.0).to(self.device)
        
        # Caricamento Pesi Baseline (Transfer Learning)
        if 'main_weights' in self.cfg:
            ckpt_path = self.cfg['main_weights']
            if os.path.exists(ckpt_path):
                print(f"Loading weights from Baseline: {ckpt_path}")
                state = torch.load(ckpt_path, map_location=self.device)
                # Pulizia chiavi per compatibilità torch.compile -> standard
                clean_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
                self.model.load_state_dict(clean_state)
            else:
                print(f"[WARNING] Weights not found at {ckpt_path}. Training from scratch/ImageNet.")

        # --- 3. TORCH COMPILE (Il vero Turbo) ---
        try:
            print("Compiling Refiner with torch.compile()...")
            self.model = torch.compile(self.model)
        except Exception as e:
            print(f"Compile failed: {e}. Continuing standard mode.")

        # --- 4. DATASET HIGH NOISE (Simulation) ---
        print("Loading High-Noise Dataset for Refinement Training...")
        # noise_factor=0.20 -> Simula errori fino al 20% della dimensione oggetto (copre i 4cm)
        self.train_ds = LineModPoseDataset(
            self.cfg['split_train'], self.cfg['dataset_root'], mode='train', 
            noise_factor=0.20 
        )
        
        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True, 
            num_workers=12, # Usa tutte le CPU della A100
            pin_memory=True,
            persistent_workers=True # Tiene i worker vivi
        )
        
        self.criterion = DenseFusionLoss(self.device)
        self.models_tensor = self._load_3d_models_tensor()
        
        # LR più basso della baseline perché stiamo facendo fine-tuning delicato
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['lr'])
        self.scaler = torch.cuda.amp.GradScaler() # Mixed Precision

    def _load_3d_models_tensor(self):
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
                # Sampling
                if pts.shape[0] > num_pts:
                    idx = np.random.choice(pts.shape[0], num_pts, replace=False)
                    pts = pts[idx, :]
                all_models[obj_id] = torch.from_numpy(pts).float()
        return all_models.to(self.device)

    def train(self):
        print(f"Starting Refiner Training ({self.cfg['epochs']} epochs)...")
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in range(self.cfg['epochs']):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Refine Ep {epoch+1}")
            ep_loss = 0
            steps = 0
            
            for batch in loop:
                # Caricamento asincrono (non_blocking=True)
                img = batch['image'].to(self.device, non_blocking=True)
                depth = batch['depth'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                gt_t = batch['translation'].to(self.device, non_blocking=True)
                gt_q = batch['quaternion'].to(self.device, non_blocking=True)
                cls_ids = batch['class_id'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Mixed Precision Forward
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # La rete deve predire la posa GT partendo da un input "croppato male" (simulato dal dataset)
                    pred_r, pred_t = self.model(img, depth, mask)
                    
                    pts = self.models_tensor[cls_ids.long()]
                    loss = self.criterion(pred_r, pred_t, gt_q, gt_t, pts, cls_ids)
                
                # Scaled Backward
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                ep_loss += loss.item()
                steps += 1
                loop.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = ep_loss / steps
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
            
            # Salvataggio Best Model
            if avg_loss < best_loss:
                best_loss = avg_loss
                path = os.path.join(self.cfg['save_dir'], "best_refiner_net.pth")
                torch.save(self.model.state_dict(), path)
                print(f" >>> New Best Refiner Saved! Loss: {best_loss:.5f}")