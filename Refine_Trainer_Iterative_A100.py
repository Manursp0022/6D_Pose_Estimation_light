import torch
import torch.optim as optim
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net 
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.DenseFusion_Loss import DenseFusionLoss 

class IterativeRefineTrainer:
    def __init__(self, config):
        self.cfg = config
        
        # --- 1. SETUP HARDWARE A100 ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            
            # [A100 MAGIC] Abilita TensorFloat-32 (TF32)
            # Aumenta drasticamente la velocità delle matmul mantenendo precisione
            torch.set_float32_matmul_precision('high') 
            
            # Ottimizzazione kernel CUDNN
            torch.backends.cudnn.benchmark = True 
            
            print(f">>> A100 DETECTED: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("WARNING: Training on CPU is not recommended.")
        
        print(f">>> Initializing Iterative Feature Refiner...")
        self.model = DenseFusion_Masked_DualAtt_Net(pretrained=False).to(self.device)
        
        if torch.cuda.is_available():
            try:
                print("Compiling model with torch.compile()...")
                # 'reduce-overhead' è ottimo per loop veloci, ma usa più memoria all'avvio
                # Se ti da OutOfMemory, togli mode='reduce-overhead'
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception as e:
                print(f"Compile warning (skipping): {e}")

        # --- 3. CARICAMENTO BASELINE ---
        if os.path.exists(self.cfg['main_weights']):
            print(f"Loading Baseline Weights: {self.cfg['main_weights']}")
            state = torch.load(self.cfg['main_weights'], map_location=self.device)
            # Pulizia nomi chiavi per compatibilità
            clean_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
            # Strict=False per ignorare i pesi mancanti del Refiner
            self.model.load_state_dict(clean_state, strict=False)
        else:
            raise FileNotFoundError(f"Baseline weights not found at {self.cfg['main_weights']}")
        
        # --- 4. FREEZE PARZIALE ---
        # Congeliamo la Baseline, alleniamo solo il Refiner
        for name, param in self.model.named_parameters():
            if 'refiner' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable Parameters (Refiner): {trainable_params}")

        # --- 5. FUSED ADAM ---
        # L'argomento 'fused=True' sposta la logica dell'optimizer sulla GPU
        self.optimizer = optim.Adam(
            self.model.refiner.parameters(), 
            lr=self.cfg.get('lr', 0.0001), 
            fused=True 
        )
        
        self.criterion = DenseFusionLoss(self.device)
        
        # --- 6. DATASET & DATALOADER ---
        print("Loading Dataset...")
        self.train_ds = LineModPoseDataset(
            self.cfg['split_train'], 
            self.cfg['dataset_root'], 
            mode='train'
        )
        
        # Dataloader ottimizzato per throughput elevato
        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True, 
            num_workers=12,         # Su Colab A100 usa 12 (max vCPU)
            pin_memory=True,        # Velocizza .to(device)
            persistent_workers=True,# Mantiene i processi vivi tra le epoche
            prefetch_factor=4,      # Precarica 4 batch per worker
            drop_last=True          # Evita batch piccoli finali che sballano le statistiche
        )

    def train(self):
        self.model.eval() # Baseline in Eval
        self.model.refiner.train() # Refiner in Train
        
        best_loss = float('inf')
        save_path = os.path.join(self.cfg['save_dir'], "best_iterative_refiner.pth")
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        print(f"Starting A100 Training for {self.cfg['epochs']} epochs...")
        
        for epoch in range(self.cfg['epochs']):
            loop = tqdm(self.train_loader, desc=f"Ep {epoch+1}", mininterval=1.0)
            ep_loss = 0.0
            steps = 0
            
            for batch in loop:
                # --- A. CARICAMENTO ASINCRONO ---
                img = batch['image'].to(self.device, non_blocking=True)
                depth = batch['depth'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                gt_r = batch['quaternion'].to(self.device, non_blocking=True)
                gt_t = batch['translation'].to(self.device, non_blocking=True)
                points = batch['points'].to(self.device, non_blocking=True) 
                obj_ids = batch['class_id'].to(self.device, non_blocking=True)
                
                # 'set_to_none=True' è leggermente più veloce di 'zero_grad()'
                self.optimizer.zero_grad(set_to_none=True)
                
                # --- B. BFLOAT16 (Specifico per A100) ---
                # A100 supporta bfloat16 che ha lo stesso range di float32.
                # Non serve GradScaler! È più stabile di float16.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # refine_iters=3: La rete prova a correggersi 3 volte
                    pred_r, pred_t = self.model(img, depth, mask, refine_iters=3)
                    loss = self.criterion(pred_r, pred_t, gt_r, gt_t, points, obj_ids)
                
                # Backward diretta (senza scaler)
                loss.backward()
                self.optimizer.step()
                
                ep_loss += loss.item()
                steps += 1
                
                # Aggiorniamo la barra meno spesso per guadagnare qualche ms
                if steps % 10 == 0:
                    loop.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = ep_loss / steps
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.5f}")
            
            # Salvataggio intelligente
            if avg_loss < best_loss:
                print(f" >>> 💾 New Best Model! ({best_loss:.4f} -> {avg_loss:.4f})")
                best_loss = avg_loss
                torch.save(self.model.state_dict(), save_path)
            
        print("Training Completed.")