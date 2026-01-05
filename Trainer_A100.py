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
#from utils.Posenet_utils.posenet_dataset_Alt import LineModPoseDataset_Alt
from utils.Posenet_utils.posenet_dataset_ALLMasked import LineModPoseDatasetMasked
from utils.Posenet_utils.posenet_dataset_AltMasked import LineModPoseDataset_AltMasked
from utils.Posenet_utils.DenseFusion_Loss_log import DenseFusionLoss

from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net

from models.DFMasked_DualAtt_NetVar import DenseFusion_Masked_DualAtt_NetVar
from models.DFMasked_DualAtt_NetVarGlobal import DenseFusion_Masked_DualAtt_NetVarGlobal

from models.DFMasked_DualAtt_NetVar_Weighted_WRefiner import DenseFusion_Masked_DualAtt_NetVarWRef
from models.DFMasked_DualAtt_NetVarGlobal_WRefiner import DenseFusion_Masked_DualAtt_NetVarGlobal_WRef

class DAMFTurboTrainerA100:
    def __init__(self, config):
        self.cfg = config
        
        self.device = self._get_device()
 
        os.makedirs(self.cfg['save_dir'], exist_ok=True)

        # MIXED PRECISION SCALER ---
        self.scaler = torch.cuda.amp.GradScaler()

        print("Initializing  DenseFusion_Masked_DualAtt_NetVar (A100 Optimized)...")
        self.model = DenseFusion_Masked_DualAtt_NetVar(
            pretrained=True, 
            temperature=self.cfg['temperature']
        ).to(self.device)

        """
        self.model = DenseFusion_Masked_DualAtt_NetVarGlobal(
            pretrained=True, 
        ).to(self.device)
        """

        """
        self.model = DenseFusion_Masked_DualAtt_NetVarWRef(
            pretrained=True, 
            temperature=self.cfg['temperature']
        ).to(self.device)
        """

        if 'resume_from' in self.cfg and self.cfg['resume_from'] is not None:
            print(f"🔄 FINE-TUNING MODE: Loading weights from {self.cfg['resume_from']}")
            if os.path.exists(self.cfg['resume_from']):
                checkpoint = torch.load(self.cfg['resume_from'], map_location=self.device)
                state_dict = self._remove_compile_prefix(checkpoint['model_state_dict'])
                
                # Carichiamo solo i pesi del modello
                # (Ignoriamo optimizer e epoch perché stiamo iniziando un nuovo stage con LR diverso)
                self.model.load_state_dict(state_dict)
                print("✅ Weights loaded successfully!")
            else:
                raise FileNotFoundError(f"Checkpoint not found at {self.cfg['resume_from']}")
        
        try:
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
        except Exception as e:
            print(f"Torch compile failed (ignore if not PyTorch 2.0): {e}")

        # B. Setup Loss, con opzione weighted o standard, weighted L = a rotLoss + b TrasLoss, Standard: AddLoss
        use_weighted = self.cfg.get('use_weighted_loss', False)
        self.criterion = DenseFusionLoss(
            self.device,
            rot_weight=self.cfg.get('rot_weight',1.0),
            trans_weight=self.cfg.get('trans_weight',0.3),
            use_weighted=use_weighted
        )
        print(f"Loss Type: {'Weighted' if use_weighted else 'Standard ADD (Paper)'}")
        self.models_tensor = self._load_3d_models_tensor()

        self.train_loader, self.val_loader = self._setup_data()
        
        self.optimizer = self._setup_optimizer()

        #self.optimizer = self._setup_separate_optimizer()
        
        # E. Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=self.cfg.get('T_0', 20), 
            T_mult=self.cfg.get('T_mult', 2), 
            eta_min=self.cfg.get('eta_min', 1e-6)
        )
        """
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,  # Decay ogni 30 epoche
            gamma=0.5      # Dimezza il LR
        )
        """
        
        # F. Tracking
        self.history = {
            'train_loss': [], 
            'val_loss': [],
            'train_rot_loss': [],
            'train_trans_loss': [],
            'val_rot_loss': [],
            'val_trans_loss': []
        }
        self.best_val_loss = float('inf')

    def _get_device(self):
        """Selezione automatica del device migliore disponibile."""
        if torch.backends.mps.is_available():
            print("✅ Using Apple MPS acceleration")
            self.num_workers = 12 
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("✅ Using CUDA")
            self.num_workers = 12 
            return torch.device("cuda")
        else:
            print("⚠️  Using CPU (slower)")
            return torch.device("cpu")
    
    def _remove_compile_prefix(self, state_dict):
        """
        Rimuove il prefisso '_orig_mod.' dai pesi salvati con torch.compile().
        
        Args:
            state_dict: State dict con o senza prefisso
            
        Returns:
            State dict pulito senza prefisso
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            # Rimuovi il prefisso '_orig_mod.' se presente
            new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
            new_state_dict[new_key] = value
        return new_state_dict


    def _setup_separate_optimizer(self):
        """
        Setup optimizer con learning rates differenziati:
        - Backbone pretrained: LR più basso (fine-tuning)
        - Heads e fusion: LR normale (training from scratch)
        """
        backbone_params = []
        head_params = []
        
        # Separa parametri backbone vs heads
        for name, param in self.model.named_parameters():
            if 'rgb_backbone' in name or 'depth_backbone' in name or 'rgb_layer' in name or 'depth_layer' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        base_lr = self.cfg.get('lr', 1e-4)
        backbone_lr = base_lr * 0.1  # 10x più lento per backbone pretrained
        
        #print(f"Optimizer Setup: Backbone LR={backbone_lr:.2e}, Heads LR={base_lr:.2e}")
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 1e-3}, # Alza weight decay qui
            {'params': head_params, 'lr': base_lr, 'weight_decay': 1e-4}
        ])
        
        return optimizer

    def _setup_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.cfg.get('lr', 1e-4), weight_decay=1e-4)

    def _setup_data(self):
        print("Loading Datasets...")

        if self.cfg['training_mode'] == "easy":
            print("Training mode set to : ",self.cfg['training_mode'])
            train_ds = LineModPoseDatasetMasked(
                self.cfg['split_train'], 
                self.cfg['dataset_root'], 
                mode='train'
            )
            val_ds = LineModPoseDatasetMasked(
                self.cfg['split_val'], 
                self.cfg['dataset_root'], 
                mode='val'
            )
        elif self.cfg['training_mode'] == "hard":
            print("Training mode set to : ",self.cfg['training_mode'])
            train_ds = LineModPoseDataset_AltMasked(
                self.cfg['dataset_root'], 
                mode='train'
            )
            val_ds = LineModPoseDataset_AltMasked( 
                self.cfg['dataset_root'], 
                mode='val'
            )

        train_loader = DataLoader(
            train_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True, 
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.cfg['batch_size'], 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        print(f"Data Loaded: {len(train_ds)} Train, {len(val_ds)} Val samples.")
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
        running_rot_loss = 0.0
        running_trans_loss = 0.0
        steps = 0
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch+1} [Train]")
        
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            depths = batch['depth'].to(self.device, non_blocking=True)
            masks  = batch['mask'].to(self.device, non_blocking=True)
            cam_params = batch['cam_params'].to(self.device, non_blocking=True)
            bb_info = batch['bbox_norm'].to(self.device, non_blocking=True)
            gt_t = batch['translation'].to(self.device, non_blocking=True)
            gt_q = batch['quaternion'].to(self.device, non_blocking=True)
            class_ids = batch['class_id'].to(self.device)

            self.optimizer.zero_grad()

            # --- AUTOMATIC MIXED PRECISION ---
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # CORRETTO: rimosso return_debug
                pred_rot, pred_trans = self.model(images, depths,bb_info,cam_params, mask=masks)
                
                current_model_points = self.models_tensor[class_ids.long()] 
                loss, metrics = self.criterion(
                    pred_rot, pred_trans, gt_q, gt_t, 
                    current_model_points, class_ids,
                    return_metrics=True
                )

            # Scaled Backward Pass
            self.scaler.scale(loss).backward() #scaler multiplies the loss by a large number (e.g. x1000) before calculating the gradients (.scale(loss).backward()). This makes the gradients ‘manageable’ numbers.
            
            # Gradient clipping per stabilità
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            running_rot_loss += metrics['rot_loss']
            running_trans_loss += metrics['trans_loss']
            steps += 1
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Rot': f"{metrics['rot_loss']:.4f}",
                'Trans': f"{metrics['trans_loss']:.4f}"
            })

        avg_loss = running_loss / steps
        avg_rot = running_rot_loss / steps
        avg_trans = running_trans_loss / steps
        
        return avg_loss, avg_rot, avg_trans

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        running_rot_loss = 0.0
        running_trans_loss = 0.0
        steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="[Val]"):
                images = batch['image'].to(self.device, non_blocking=True)
                depths = batch['depth'].to(self.device, non_blocking=True)
                masks  = batch['mask'].to(self.device, non_blocking=True)
                cam_params = batch['cam_params'].to(self.device, non_blocking=True)
                bb_info = batch['bbox_norm'].to(self.device, non_blocking=True)
                gt_t = batch['translation'].to(self.device, non_blocking=True)
                gt_q = batch['quaternion'].to(self.device, non_blocking=True)
                class_ids = batch['class_id'].to(self.device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_rot, pred_trans = self.model(images, depths,bb_info,cam_params, mask=masks)
                    
                    current_model_points = self.models_tensor[class_ids.long()]
                    loss, metrics = self.criterion(
                        pred_rot, pred_trans, gt_q, gt_t,
                        current_model_points, class_ids,
                        return_metrics=True
                    )

                running_loss += loss.item()
                running_rot_loss += metrics['rot_loss']
                running_trans_loss += metrics['trans_loss']
                steps += 1
        
        avg_loss = running_loss / steps
        avg_rot = running_rot_loss / steps
        avg_trans = running_trans_loss / steps
        
        return avg_loss, avg_rot, avg_trans

    def run(self):
        print(f"Starting A100 TURBO Training ({self.cfg['epochs']} epochs)...")
        loss_type = "Weighted Loss" if self.criterion.use_weighted else "Standard ADD Loss (Paper)"
        print(f"Using: {loss_type}")
        if self.criterion.use_weighted:
            print(f"Weights: Rotation={self.criterion.rot_weight}, Translation={self.criterion.trans_weight}")
        
        early_stop_counter = 0
        patience_limit = self.cfg.get('early_stop_patience', 20)
        
        for epoch in range(self.cfg['epochs']):
            train_loss, train_rot, train_trans = self.train_epoch(epoch)
            val_loss, val_rot, val_trans = self.validate()
            
            self.scheduler.step()
            
            # Tracking
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rot_loss'].append(train_rot)
            self.history['train_trans_loss'].append(train_trans)
            self.history['val_rot_loss'].append(val_rot)
            self.history['val_trans_loss'].append(val_trans)
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.cfg['epochs']} | LR: {current_lr:.2e}")
            print(f"Train - Total: {train_loss:.4f} | Rot: {train_rot:.4f} | Trans: {train_trans:.4f}")
            print(f"Val   - Total: {val_loss:.4f} | Rot: {val_rot:.4f} | Trans: {val_trans:.4f}")
            print(f"{'='*60}\n")

            # Model Saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                early_stop_counter = 0
                
                path = os.path.join(self.cfg['save_dir'], 'DenseFusion_Masked_DualAtt_NetVar_WOAttention.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_rot_loss': val_rot,
                    'val_trans_loss': val_trans,
                    'config': self.cfg
                }, path)
                
                print(f"✓ New Best Model Saved! Val Loss: {val_loss:.5f}")
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter} epochs")
                
            if early_stop_counter >= patience_limit:
                print("\n!!! Early Stopping Triggered !!!")
                break
        
        self.plot_results()
    
    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Val', color='orange')
        axes[0, 0].set_title('Total Loss (Weighted)')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Rotation Loss
        axes[0, 1].plot(self.history['train_rot_loss'], label='Train Rot', color='green')
        axes[0, 1].plot(self.history['val_rot_loss'], label='Val Rot', color='red')
        axes[0, 1].set_title('Rotation Loss')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('ADD (meters)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Translation Loss
        axes[1, 0].plot(self.history['train_trans_loss'], label='Train Trans', color='purple')
        axes[1, 0].plot(self.history['val_trans_loss'], label='Val Trans', color='brown')
        axes[1, 0].set_title('Translation Loss')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('L2 Distance (meters)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Loss Comparison
        axes[1, 1].plot(self.history['val_rot_loss'], label='Rotation', color='red')
        axes[1, 1].plot(self.history['val_trans_loss'], label='Translation', color='brown')
        axes[1, 1].set_title('Val: Rotation vs Translation')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        path = os.path.join(self.cfg['save_dir'], 'training_analysis.png')
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"Training plots saved to {path}")