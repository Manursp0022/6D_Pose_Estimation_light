import torch
import torch.optim as optim
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from plyfile import PlyData 

from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net 
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.DenseFusion_Loss import DenseFusionLoss 

class IterativeRefineTrainer:
    def __init__(self, config):
        self.cfg = config
        
        # --- 1. SETUP HARDWARE A100 ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.set_float32_matmul_precision('high') 
            torch.backends.cudnn.benchmark = True 
            print(f">>> A100 DETECTED: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
        
        print(f">>> Initializing Iterative Feature Refiner...")
        self.model = DenseFusion_Masked_DualAtt_Net(pretrained=False).to(self.device)
        
        # --- 2. LOAD 3D MODELS (FIX PRESTAZIONI) ---
        # Carichiamo ORA, una volta sola, in VRAM. Non nel loop!
        self.models_tensor = self._load_3d_models_tensor().to(self.device)

        if torch.cuda.is_available():
            try:
                print("Compiling model with torch.compile()...")
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception as e:
                print(f"Compile warning (skipping): {e}")

        # --- 3. CARICAMENTO BASELINE ---
        if os.path.exists(self.cfg['main_weights']):
            print(f"Loading Baseline Weights: {self.cfg['main_weights']}")
            state = torch.load(self.cfg['main_weights'], map_location=self.device)
            clean_state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
            self.model.load_state_dict(clean_state, strict=False)
        else:
            raise FileNotFoundError(f"Baseline weights not found at {self.cfg['main_weights']}")
        
        # --- 4. FREEZE PARZIALE ---
        for name, param in self.model.named_parameters():
            if 'refiner' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable Parameters (Refiner): {trainable_params}")

        # --- 5. FUSED ADAM ---
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
        
        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True, 
            num_workers=12,         
            pin_memory=True,        
            persistent_workers=True,
            prefetch_factor=4,      
            drop_last=True          
        )

    def _load_3d_models_tensor(self):
        print("Loading 3D Models into VRAM...")
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        max_id = 16 
        # Usa il valore dalla config o default a 500
        num_pts = self.cfg.get('num_points_mesh', 500) 
        all_models = torch.zeros((max_id, num_pts, 3), dtype=torch.float32)
        
        obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        for obj_id in obj_ids:
            path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
            if os.path.exists(path):
                ply = PlyData.read(path)
                v = ply['vertex']
                pts = np.stack([v['x'], v['y'], v['z']], axis=-1) 
                
                # Check unità di misura (mm -> m)
                if np.mean(np.abs(pts)) > 10.0:
                    pts = pts / 1000.0
                    
                # Sampling
                if pts.shape[0] > num_pts:
                    idx = np.random.choice(pts.shape[0], num_pts, replace=False)
                    pts = pts[idx, :]
                all_models[obj_id] = torch.from_numpy(pts).float()
            else:
                print(f"[WARN] Mesh not found: {path}")
        return all_models # Restituisce CPU tensor, poi .to(device) in init

    def train(self):
        self.model.eval() 
        self.model.refiner.train()
        
        best_loss = float('inf')
        save_path = os.path.join(self.cfg['save_dir'], "best_iterative_refiner.pth")
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        print(f"Starting A100 Training for {self.cfg['epochs']} epochs...")
        
        for epoch in range(self.cfg['epochs']):
            loop = tqdm(self.train_loader, desc=f"Ep {epoch+1}", mininterval=1.0)
            ep_loss = 0.0
            steps = 0
            
            for batch in loop:
                img = batch['image'].to(self.device, non_blocking=True)
                depth = batch['depth'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                gt_r = batch['quaternion'].to(self.device, non_blocking=True)
                gt_t = batch['translation'].to(self.device, non_blocking=True)
                obj_ids = batch['class_id'].to(self.device, non_blocking=True)
                
                # --- FIX CRITICO ---
                # Peschiamo i punti corretti dalla memoria GPU usando gli ID del batch
                # Shape risultante: [Batch_Size, 500, 3]
                points = self.models_tensor[obj_ids.long()]
                
                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_r, pred_t = self.model(img, depth, mask, refine_iters=3)
                    loss = self.criterion(pred_r, pred_t, gt_r, gt_t, points, obj_ids)
                
                loss.backward()
                self.optimizer.step()
                
                ep_loss += loss.item()
                steps += 1
                
                if steps % 10 == 0:
                    loop.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_loss = ep_loss / steps
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.5f}")
            
            if avg_loss < best_loss:
                print(f" >>> 💾 New Best Model! ({best_loss:.4f} -> {avg_loss:.4f})")
                best_loss = avg_loss
                torch.save(self.model.state_dict(), save_path)
            
        print("Training Completed.")