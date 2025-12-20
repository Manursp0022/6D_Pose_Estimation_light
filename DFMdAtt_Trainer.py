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

from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net
from utils.Posenet_utils.DenseFusion_Loss import DenseFusionLoss

class DFMdAtt_Trainer:
    def __init__(self, config):
        self.cfg = config
        
        # Setup Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.num_workers = 8
            print(">>> Using CUDA (NVIDIA)")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.num_workers = 0 
            print(">>> Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            self.num_workers = 0
            print(">>> Using CPU")

        os.makedirs(self.cfg['save_dir'], exist_ok=True)

        # A. Setup Modello
        print("Initializing DenseFusion TURBO Net...")
        self.model = DenseFusion_Masked_DualAtt_Net(
            pretrained=True, 
            temperature=self.cfg.get('temperature', 2.0)
        ).to(self.device)
        
        # B. Setup Loss & Dati 3D
        self.criterion = DenseFusionLoss(self.device)
        self.models_tensor = self._load_3d_models_tensor()

        # C. Setup Dataset (Train & Val)
        self.train_loader, self.val_loader = self._setup_data()
        
        # D. Optimizer & Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['lr'])
        
        # Riduce LR se la Val Loss non scende per 'patience' epoche
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=self.cfg['scheduler_patience'], verbose=True
        )
        
        self.history = {'train_loss': [], 'val_loss': []}

    def _setup_data(self):
        print("Loading Datasets...")
        train_ds = LineModPoseDataset(self.cfg['split_train'], self.cfg['dataset_root'], mode='train')
        val_ds = LineModPoseDataset(self.cfg['split_val'], self.cfg['dataset_root'], mode='val')
        
        train_loader = DataLoader(train_ds, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_ds, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=self.num_workers)
        
        print(f"Data Loaded: {len(train_ds)} Train samples, {len(val_ds)} Val samples.")
        return train_loader, val_loader

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
            images = batch['image'].to(self.device)
            depths = batch['depth'].to(self.device)
            masks  = batch['mask'].to(self.device)
            
            gt_t = batch['translation'].to(self.device)
            gt_q = batch['quaternion'].to(self.device)
            class_ids = batch['class_id'].to(self.device)

            self.optimizer.zero_grad()

            # Forward (No Debug info)
            pred_rot, pred_trans = self.model(images, depths, mask=masks, return_debug=False)

            current_model_points = self.models_tensor[class_ids.long()] 
            loss = self.criterion(pred_rot, pred_trans, gt_q, gt_t, current_model_points, class_ids)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            steps += 1
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        return running_loss / steps

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        steps = 0
        
        # Nessun gradiente in validazione
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="[Val]"):
                images = batch['image'].to(self.device)
                depths = batch['depth'].to(self.device)
                masks  = batch['mask'].to(self.device)
                
                gt_t = batch['translation'].to(self.device)
                gt_q = batch['quaternion'].to(self.device)
                class_ids = batch['class_id'].to(self.device)

                pred_rot, pred_trans = self.model(images, depths, mask=masks, return_debug=False)

                current_model_points = self.models_tensor[class_ids.long()]
                loss = self.criterion(pred_rot, pred_trans, gt_q, gt_t, current_model_points, class_ids)

                running_loss += loss.item()
                steps += 1
        
        return running_loss / steps

    def run(self):
        print(f"Starting FINAL Training ({self.cfg['epochs']} epochs)...")
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        patience_limit = self.cfg.get['early_stop_patience']
        
        for epoch in range(self.cfg['epochs']):
            # 1. Train
            train_loss = self.train_epoch(epoch)
            
            # 2. Validate
            val_loss = self.validate()
            
            # 3. Scheduler Step
            # (Il ReduceLROnPlateau guarda la val_loss per decidere)
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 4. Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                path = os.path.join(self.cfg['save_dir'], 'best_turbo_model.pth')
                torch.save(self.model.state_dict(), path)
                print(f" >>> New Best Model Saved! Val Loss: {best_val_loss:.5f} <<<")
            else:
                early_stop_counter += 1
                print(f" No improvement. Early Stop Counter: {early_stop_counter}/{patience_limit}")
                
            if early_stop_counter >= patience_limit:
                print("\n!!! Early Stopping Triggered !!!")
                break
        
        print("Training Finished.")
        self.plot_results()

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train ADD Loss', color='blue')
        plt.plot(self.history['val_loss'], label='Val ADD Loss', color='orange')
        plt.title('Training Progress (ADD Metric)')
        plt.xlabel('Epochs')
        plt.ylabel('ADD Loss (avg meters)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        path = os.path.join(self.cfg['save_dir'], 'loss_curve.png')
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.show()