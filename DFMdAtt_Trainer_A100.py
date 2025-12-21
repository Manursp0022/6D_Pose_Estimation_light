import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from plyfile import PlyData

# --- IMPORT TUOI MODULI ---
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net 
from utils.Posenet_utils.DenseFusion_Loss import DenseFusionLoss

class DFTurboTrainerA100:
    def __init__(self, config):
        self.cfg = config
        
        # --- 1. OTTIMIZZAZIONE HARDWARE ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # A100 su Colab ha solitamente 12 vCPU. Usiamole tutte.
            self.num_workers = 12 
            # Benchmark trova l'algoritmo di convoluzione più veloce per la tua dimensione input
            torch.backends.cudnn.benchmark = True 
            print(f">>> A100 DETECTED. Using {torch.cuda.get_device_name(0)}")
            print(f">>> Num Workers set to {self.num_workers}")
        else:
            self.device = torch.device("cpu")
            self.num_workers = 0

        os.makedirs(self.cfg['save_dir'], exist_ok=True)

        # --- 2. MIXED PRECISION SCALER ---
        # Fondamentale per A100: gestisce la precisione float16 senza perdere gradienti
        self.scaler = torch.cuda.amp.GradScaler()

        # A. Setup Modello
        print("Initializing DenseFusion TURBO Net (A100 Optimized)...")
        self.model = DenseFusion_Masked_DualAtt_Net(
            pretrained=True, 
            temperature=self.cfg.get('temperature', 2.0)
        ).to(self.device)
        
        # --- 3. TORCH COMPILE (PyTorch 2.0+) ---
        # Compila il modello per ottimizzare i kernel CUDA
        try:
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
        except Exception as e:
            print(f"Torch compile failed (ignore if not PyTorch 2.0): {e}")

        # B. Setup Loss & Dati 3D
        self.criterion = DenseFusionLoss(self.device)
        self.models_tensor = self._load_3d_models_tensor()

        # C. Setup Dataset
        self.train_loader, self.val_loader = self._setup_data()
        
        # D. Optimizer & Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=self.cfg['scheduler_patience'], verbose=True
        )
        
        self.history = {'train_loss': [], 'val_loss': []}

    def _setup_data(self):
        print("Loading Datasets...")
        train_ds = LineModPoseDataset(self.cfg['split_train'], self.cfg['dataset_root'], mode='train')
        val_ds = LineModPoseDataset(self.cfg['split_val'], self.cfg['dataset_root'], mode='val')
        
        # --- 4. PIN MEMORY ---
        # pin_memory=True velocizza il trasferimento RAM -> VRAM
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True, 
            persistent_workers=True # Tiene i worker vivi tra le epoche
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        print(f"Data Loaded: {len(train_ds)} Train samples, {len(val_ds)} Val samples.")
        return train_loader, val_loader

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

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        steps = 0
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch+1} [Train]")
        
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            depths = batch['depth'].to(self.device, non_blocking=True)
            masks  = batch['mask'].to(self.device, non_blocking=True)
            gt_t = batch['translation'].to(self.device, non_blocking=True)
            gt_q = batch['quaternion'].to(self.device, non_blocking=True)
            class_ids = batch['class_id'].to(self.device)

            self.optimizer.zero_grad()

            # --- 5. AUTOMATIC MIXED PRECISION (AMP) ---
            # Esegue il forward pass in float16 dove possibile
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_rot, pred_trans = self.model(images, depths, mask=masks, return_debug=False)
                
                current_model_points = self.models_tensor[class_ids.long()] 
                loss = self.criterion(pred_rot, pred_trans, gt_q, gt_t, current_model_points, class_ids)

            # Scaled Backward Pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            steps += 1
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        return running_loss / steps

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="[Val]"):
                images = batch['image'].to(self.device, non_blocking=True)
                depths = batch['depth'].to(self.device, non_blocking=True)
                masks  = batch['mask'].to(self.device, non_blocking=True)
                gt_t = batch['translation'].to(self.device, non_blocking=True)
                gt_q = batch['quaternion'].to(self.device, non_blocking=True)
                class_ids = batch['class_id'].to(self.device)

                # Anche in validazione usiamo autocast per coerenza e velocità
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_rot, pred_trans = self.model(images, depths, mask=masks, return_debug=False)
                    current_model_points = self.models_tensor[class_ids.long()]
                    loss = self.criterion(pred_rot, pred_trans, gt_q, gt_t, current_model_points, class_ids)

                running_loss += loss.item()
                steps += 1
        
        return running_loss / steps

    def run(self):
        print(f"Starting A100 TURBO Training ({self.cfg['epochs']} epochs)...")
        best_val_loss = float('inf')
        early_stop_counter = 0
        patience_limit = self.cfg['early_stop_patience']
        
        for epoch in range(self.cfg['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                path = os.path.join(self.cfg['save_dir'], 'best_turbo_model_A100.pth')
                torch.save(self.model.state_dict(), path)
                print(f" >>> New Best Model Saved! Val Loss: {best_val_loss:.5f} <<<")
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience_limit:
                print("\n!!! Early Stopping Triggered !!!")
                break
        
        self.plot_results()
    
    def plot_results(self):
        # (Codice plot uguale a prima)
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train ADD Loss', color='blue')
        plt.plot(self.history['val_loss'], label='Val ADD Loss', color='orange')
        plt.title('Training Progress (ADD Metric)')
        plt.xlabel('Epochs')
        plt.ylabel('ADD Loss (avg meters)')
        plt.legend()
        plt.grid(True)
        path = os.path.join(self.cfg['save_dir'], 'loss_curve.png')
        plt.savefig(path)
        plt.show()