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
            self.num_workers = 0 # MPS non ama multiprocessing pesante
            print(">>> Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            self.num_workers = 0
            print(">>> Using CPU")

        os.makedirs(self.cfg['save_dir'], exist_ok=True)

        # A. Setup Modello Turbo
        print("Initializing DenseFusion TURBO Net...")
        self.model = DenseFusion_Masked_DualAtt_Net(
            pretrained=True, 
            temperature=self.cfg.get('temperature', 2.0) # Temperatura Confidence
        ).to(self.device)
        
        # B. Setup Loss & Dati 3D
        self.criterion = DenseFusionLoss(self.device)
        self.models_tensor = self._load_3d_models_tensor() # Carica geometria per la Loss

        # C. Setup Dataset
        self.train_loader = self._setup_data()
        
        # D. Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['lr'])
        self.history = {'train_loss': []}

    def _setup_data(self):
        print("Loading Dataset...")
        train_ds = LineModPoseDataset(self.cfg['split_train'], self.cfg['dataset_root'], mode='train')
        return DataLoader(train_ds, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=self.num_workers)

    def _load_3d_models_tensor(self):
        print("Loading 3D Models (Vertices) into VRAM for Loss Calculation...")
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        max_id = 16 
        num_pts = self.cfg['num_points_mesh']
        # Tensore unico [16, 500, 3]
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
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch+1}")
        
        for batch in pbar:
            # Inputs
            images = batch['image'].to(self.device)
            depths = batch['depth'].to(self.device)
            masks  = batch['mask'].to(self.device) # <--- La Maschera!
            
            # GT Labels
            gt_translation = batch['translation'].to(self.device)
            gt_quaternion = batch['quaternion'].to(self.device)
            class_ids = batch['class_id'].to(self.device)

            self.optimizer.zero_grad()

            # --- FORWARD PASS (Con Diagnostica) ---
            pred_rot, pred_trans, debug_stats = self.model(
                images, depths, mask=masks, return_debug=True
            )

            # --- LOSS CALCULATION (SOTA ADD) ---
            # Recupera i punti dell'oggetto corretto dal tensore precaricato
            current_model_points = self.models_tensor[class_ids.long()] 
            
            loss = self.criterion(pred_rot, pred_trans, gt_quaternion, gt_translation, current_model_points, class_ids)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            steps += 1
            
            # Update barra (mostra la loss in cm/metri)
            pbar.set_postfix({'ADD Loss': f"{loss.item():.4f}"})

            # --- STAMPA INTELLIGENTE DI DIAGNOSTICA ---
            if steps % 50 == 0:
                tqdm.write(f"\n --- DIAGNOSTICA BATCH {steps} (Ep {epoch+1}) ---")
                
                # Attenzione (Geometrica)
                att_msg = (f" [Attention] Mean: {debug_stats['att_mean']:.3f} | "
                           f"Max: {debug_stats['att_max']:.3f} | "
                           f"Min: {debug_stats['att_min']:.3f}")
                tqdm.write(att_msg)
                
                # Confidence (Pesi Pixel)
                conf_msg = (f" [Confidence] Max Peak: {debug_stats['conf_max']:.4f} (Ideal: >0.1) | "
                            f"Std: {debug_stats['conf_std']:.4f}")
                tqdm.write(conf_msg)
                
                # Warning Temperature
                if debug_stats['conf_max'] > 0.95:
                    tqdm.write(" [WARNING] Confidence Peak ~1.0. La rete si fida di UN solo pixel. Alza la Temperature!")
                if debug_stats['conf_max'] < 0.025:
                    tqdm.write(" [INFO] Confidence piatta (Inizio training).")
                
                tqdm.write(" ---------------------------------")

        return running_loss / steps

    def run(self):
        print(f"Starting TURBO Training ({self.cfg['epochs']} epochs)...")
        best_loss = float('inf')
        
        for epoch in range(self.cfg['epochs']):
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            print(f"Epoch {epoch+1} Completed. Mean ADD Loss: {train_loss:.4f}")

            # Salviamo sempre il best model
            if train_loss < best_loss:
                best_loss = train_loss
                path = os.path.join(self.cfg['save_dir'], 'best_turbo_model.pth')
                torch.save(self.model.state_dict(), path)
                print(f"----> New Best Model Saved! Loss: {best_loss:.5f} <----")