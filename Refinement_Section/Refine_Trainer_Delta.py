import torch
import torch.optim as optim
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from plyfile import PlyData
import numpy as np

# IMPORT
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net 
from utils.Posenet_utils.delta_utils import compute_delta_target

class RefineTrainerDelta:
    def __init__(self, config):
        self.cfg = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True 
            torch.set_float32_matmul_precision('high')
            print(f">>> A100 DETECTED: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
        
        # MODELLO (Stessa architettura, ma impara i Delta)
        print("Initializing Delta Refiner (CosyPose Logic)...")
        self.model = DenseFusion_Masked_DualAtt_Net(pretrained=True, temperature=2.0).to(self.device)
        
        # Carica pesi della Baseline (Transfer Learning)
        # Fondamentale: il refiner deve "vedere" come la baseline per capire le feature
        if 'main_weights' in self.cfg and os.path.exists(self.cfg['main_weights']):
            print(f"Loading baseline weights from: {self.cfg['main_weights']}")
            state = torch.load(self.cfg['main_weights'], map_location=self.device)
            state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
            self.model.load_state_dict(state)
        
        # Compilazione (Velocità +30%)
        try:
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
        except Exception as e:
            print(f"Compile failed: {e}")

        # 2. DATASET
        # Usiamo noise_factor=0.0 nel dataset perché generiamo il rumore dinamicamente in GPU
        self.train_ds = LineModPoseDataset(
            self.cfg['split_train'], self.cfg['dataset_root'], mode='train', 
            noise_factor=0.0 
        )
        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True, 
            num_workers=12, # Usa tutte le CPU della A100
            pin_memory=True
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg['lr'])
        self.scaler = torch.cuda.amp.GradScaler() # Mixed Precision
        self.criterion = torch.nn.L1Loss() # L1 è più robusta per i Delta

    def generate_noisy_pose(self, gt_t, gt_r):
        """
        Genera una posa 'start' sbagliata partendo dalla GT.
        Simula l'errore che commetterebbe la Baseline.
        """
        # Rumore Traslazione (es. +/- 10% dell'oggetto o fisso in metri)
        # Qui usiamo un mix: errore relativo + assoluto
        noise_t = (torch.rand_like(gt_t) - 0.5) * 0.10 # +/- 5-10 cm circa
        start_t = gt_t + noise_t
        
        # Rumore Rotazione
        noise_r = (torch.rand_like(gt_r) - 0.5) * 0.10
        start_r = gt_r + noise_r
        start_r = torch.nn.functional.normalize(start_r, p=2, dim=1)
        
        return start_t, start_r

    def train(self):
        print(f"Starting Delta Training ({self.cfg['epochs']} epochs)...")
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        for epoch in range(self.cfg['epochs']):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Ep {epoch+1}")
            ep_loss = 0
            
            for batch in loop:
                img = batch['image'].to(self.device, non_blocking=True)
                depth = batch['depth'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                
                gt_t = batch['translation'].to(self.device, non_blocking=True)
                gt_q = batch['quaternion'].to(self.device, non_blocking=True)
                
                # 1. Generiamo Posa Iniziale Sbagliata (Simulazione Baseline)
                # In CosyPose, si farebbe il crop su questa posa.
                # Qui, per efficienza estrema, assumiamo che l'immagine in input (centrata su GT)
                # sia visualmente simile a un crop centrato su 'start_t' se l'errore è piccolo.
                start_t, start_q = self.generate_noisy_pose(gt_t, gt_q)
                
                # 2. Calcoliamo il Target Delta
                # La rete deve predire: "Come vado da Start a GT?"
                target_delta_t, target_delta_r = compute_delta_target(start_t, start_q, gt_t, gt_q)
                
                self.optimizer.zero_grad()
                
                # Mixed Precision Forward
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_delta_r, pred_delta_t = self.model(img, depth, mask)
                    
                    # Loss diretta sui Delta
                    loss_t = self.criterion(pred_delta_t, target_delta_t)
                    loss_r = self.criterion(pred_delta_r, target_delta_r)
                    loss = loss_t + loss_r 

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                ep_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = ep_loss/len(self.train_loader)
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
            
            # Salvataggio
            torch.save(self.model.state_dict(), os.path.join(self.cfg['save_dir'], "best_delta_refiner.pth"))