import torch
import torch.optim as optim
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

# IMPORT DEI TUOI MODELLI
from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net 
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.DenseFusion_Loss import DenseFusionLoss 

class IterativeRefineTrainer:
    def __init__(self, config):
        self.cfg = config
        
        # --- 1. H100 SETUP ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            
            # A. ABILITA TF32 (Fondamentale su Ampere/Hopper)
            # Permette operazioni matmul molto più veloci sui Tensor Core
            torch.set_float32_matmul_precision('high') 
            
            # B. CUDNN BENCHMARK
            torch.backends.cudnn.benchmark = True 
            
            print(f">>> H100 DETECTED: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
        
        print(f">>> Initializing Iterative Feature Refiner...")
        self.model = DenseFusion_Masked_DualAtt_Net(pretrained=False).to(self.device)
        
        # --- 2. COMPILE (Mode: Reduce-Overhead) ---
        # Su H100 'max-autotune' è potente ma lento all'avvio. 'reduce-overhead' è ottimo per training loop veloci.
        if torch.cuda.is_available():
            try:
                print("Compiling model with torch.compile(mode='reduce-overhead')...")
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception as e:
                print(f"Compile warning: {e}")

        # CARICAMENTO BASELINE
        if os.path.exists(self.cfg['main_weights']):
            print(f"Loading Baseline Weights: {self.cfg['main_weights']}")
            state = torch.load(self.cfg['main_weights'], map_location=self.device)
            clean_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
            self.model.load_state_dict(clean_state, strict=False)
        else:
            raise FileNotFoundError(f"Baseline weights not found at {self.cfg['main_weights']}")
        
        # FREEZE
        for name, param in self.model.named_parameters():
            if 'refiner' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # --- 3. FUSED OPTIMIZER ---
        # fused=True sposta tutto il calcolo dell'optimizer sulla GPU.
        # È molto più veloce su modelli con molti parametri o training veloci.
        self.optimizer = optim.Adam(
            self.model.refiner.parameters(), 
            lr=self.cfg.get('lr', 0.0001), 
            fused=True 
        )
        
        self.criterion = DenseFusionLoss(self.device)
        
        # --- 4. DATALOADER OTTIMIZZATO ---
        print("Loading Dataset...")
        self.train_ds = LineModPoseDataset(
            self.cfg['split_train'], 
            self.cfg['dataset_root'], 
            mode='train'
        )
        
        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True, 
            num_workers=12,         # H100 su Colab ha tante CPU, usale
            pin_memory=True,        # Fondamentale per trasferimento veloce RAM -> VRAM
            persistent_workers=True,# Non uccide i worker a fine epoca (risparmia tempo)
            prefetch_factor=4,      # Carica 4 batch in anticipo per non far aspettare la GPU
            drop_last=True
        )

    def train(self):
        self.model.eval() 
        self.model.refiner.train()
        
        best_loss = float('inf')
        save_path = os.path.join(self.cfg['save_dir'], "best_iterative_refiner.pth")
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        print(f"Starting H100 Training for {self.cfg['epochs']} epochs...")
        
        for epoch in range(self.cfg['epochs']):
            # Usiamo tqdm ma attenzione che su H100 potrebbe rallentare se stampa troppo spesso
            loop = tqdm(self.train_loader, desc=f"Ep {epoch+1}", mininterval=1.0)
            ep_loss = 0.0
            steps = 0
            
            for batch in loop:
                # --- 5. DATA TRANSFER ASINCRONO ---
                # non_blocking=True permette alla CPU di preparare il prossimo batch
                # mentre la GPU sta ancora calcolando questo.
                img = batch['image'].to(self.device, non_blocking=True)
                depth = batch['depth'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                gt_r = batch['quaternion'].to(self.device, non_blocking=True)
                gt_t = batch['translation'].to(self.device, non_blocking=True)
                points = batch['points'].to(self.device, non_blocking=True) 
                obj_ids = batch['class_id'].to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True) # set_to_none è leggermente più veloce di zero_grad
                
                # --- 6. BFLOAT16 MIXED PRECISION ---
                # Su H100, bfloat16 è IL RE. 
                # Non serve GradScaler perché bfloat16 ha lo stesso range di float32.
                # È stabile come FP32 ma veloce come FP16.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_r, pred_t = self.model(img, depth, mask, refine_iters=3)
                    loss = self.criterion(pred_r, pred_t, gt_r, gt_t, points, obj_ids)
                
                # Niente scaler.scale() necessario per BFloat16!
                loss.backward()
                self.optimizer.step()
                
                ep_loss += loss.item()
                steps += 1
                
                if steps % 10 == 0: # Aggiorna la barra meno spesso per velocità
                    loop.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = ep_loss / steps
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.5f}")
            
            if avg_loss < best_loss:
                print(f" >>> 💾 New Best Model! ({best_loss:.4f} -> {avg_loss:.4f})")
                best_loss = avg_loss
                torch.save(self.model.state_dict(), save_path)
            
        print("Training Completed.")