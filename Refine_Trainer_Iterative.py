import torch
import torch.optim as optim
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

# IMPORT DEI TUOI MODELLI
# Assicurati che i nomi dei file corrispondano agli import!
from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net 
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.DenseFusion_Loss import DenseFusionLoss 

class IterativeRefineTrainer:
    def __init__(self, config):
        self.cfg = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # A100 su Colab ha solitamente 12 vCPU. Usiamole tutte.
            self.num_workers = 12 
            # Benchmark trova l'algoritmo di convoluzione più veloce per la tua dimensione input
            torch.backends.cudnn.benchmark = True 
            print(f">>> H100/A100 DETECTED. Using {torch.cuda.get_device_name(0)}")
            print(f">>> Num Workers set to {self.num_workers}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.num_workers = 0 
            print(">>> Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            self.num_workers = 0
        
        print(f">>> Initializing Iterative Feature Refiner on {self.device}...")
        self.model = DenseFusion_Masked_DualAtt_Net(pretrained=False).to(self.device)
        if torch.cuda.is_available():
            try:
                print("Compiling model with torch.compile()...")
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"Torch compile failed (ignore if not PyTorch 2.0): {e}")
        
        # 1. CARICAMENTO BASELINE (Fondamentale!)
        if os.path.exists(self.cfg['main_weights']):
            print(f"Loading Baseline Weights: {self.cfg['main_weights']}")
            state = torch.load(self.cfg['main_weights'], map_location=self.device)
            
            # Pulizia chiavi se necessario (per compatibilità con torch.compile o DDP)
            clean_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
            
            # Carica con strict=False:
            # - Carica i pesi della Baseline (che esistono)
            # - Ignora i pesi del Refiner (che mancano nel file e saranno inizializzati random)
            self.model.load_state_dict(clean_state, strict=False)
        else:
            raise FileNotFoundError(f"Baseline weights not found at {self.cfg['main_weights']}")
        
        # 2. FREEZE DELLA BASELINE
        # Vogliamo allenare SOLO il refiner, lasciando intatta la "visione" della baseline
        for name, param in self.model.named_parameters():
            if 'refiner' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True # Assicuriamoci che il refiner sia allenabile
        
        # Verifica parametri allenabili
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable Parameters (Refiner Only): {trainable_params}")

        # 3. OPTIMIZER & LOSS
        # LR basso ma non troppo (0.0001 è standard per refinement)
        self.optimizer = optim.Adam(self.model.refiner.parameters(), lr=self.cfg.get('lr', 0.0001))
        self.criterion = DenseFusionLoss(self.device)
        
        # 4. DATASET
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
            num_workers=8,
            pin_memory=True
        )

    def train(self):
        # Baseline in Eval (per BatchNorm stabili), Refiner in Train
        self.model.eval() 
        self.model.refiner.train()
        
        best_loss = float('inf')
        save_path = os.path.join(self.cfg['save_dir'], "best_iterative_refiner.pth")
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        print(f"Starting Training for {self.cfg['epochs']} epochs...")
        
        for epoch in range(self.cfg['epochs']):
            loop = tqdm(self.train_loader, desc=f"Ep {epoch+1}/{self.cfg['epochs']}")
            ep_loss = 0.0
            steps = 0
            
            for batch in loop:
                img = batch['image'].to(self.device)
                depth = batch['depth'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                gt_r = batch['quaternion'].to(self.device)
                gt_t = batch['translation'].to(self.device)
                points = batch['points'].to(self.device) 
                obj_ids = batch['class_id'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # FORWARD CON REFINEMENT
                # refine_iters=3 significa che la rete prova a correggersi 3 volte internamente
                # Nota: qui NON passiamo 'points' alla forward perché il tuo FeatureRefiner non li usa
                pred_r, pred_t = self.model(img, depth, mask, refine_iters=3)
                
                # LOSS CALCULATION
                # La loss viene calcolata sulla posa FINALE raffinata
                loss = self.criterion(pred_r, pred_t, gt_r, gt_t, points, obj_ids)
                
                loss.backward()
                self.optimizer.step()
                
                ep_loss += loss.item()
                steps += 1
                
                loop.set_postfix(loss=f"{loss.item():.4f}")
            
            # CALCOLO MEDIA E SALVATAGGIO BEST MODEL
            avg_loss = ep_loss / steps
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.5f}")
            
            if avg_loss < best_loss:
                print(f" >>> 💾 New Best Model Found! (Loss: {best_loss:.5f} -> {avg_loss:.5f}) Saving to {save_path}")
                best_loss = avg_loss
                torch.save(self.model.state_dict(), save_path)
            
            # (Opzionale) Scheduler step qui se ne usi uno
            
        print("Training Completed.")