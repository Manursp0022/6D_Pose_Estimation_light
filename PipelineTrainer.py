import torch
import torch.optim as optim
import os
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.utils_geometric import solve_pinhole_depth
from models.RGB_D_ResNet import PoseResNetRGBD
from models.Posenet import PoseResNet
from utils.Posenet_utils.quaternion_Loss import QuaternionLoss



"""
TranslationNet + RotationNet training pipeline.
- DepthNet predicts depth from depth images.
- PoseResNet predicts rotation from RGB images.
- Translation computed via pinhole model using predicted depth and bbox.

"""
class PipelineTrainer:
    def __init__(self, config):
        """
        Inizializza il trainer con un dizionario di configurazione.
        """
        self.cfg = config
        self.device = self._get_device()
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        """self.DRS = {
            1: 102.09865663, 2: 247.50624233, 4: 172.49224865,
            5: 201.40358597, 6: 154.54551808, 8: 261.47178102,
            9: 108.99920102, 10: 164.62758848, 11: 175.88933422,
            12: 145.54287471, 13: 278.07811733, 14: 282.60129399,
            15: 212.35825148
        }"""

        self.train_loader, self.val_loader = self._setup_data()
        self.module, self.optimizer, self.scheduler = self._setup_model()
        self.criterion_trans = torch.nn.L1Loss()
        self.criterion_rot = QuaternionLoss()
        
        # Metrics
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_loss_t': [], 'val_loss_t': [],
            'train_loss_r': [], 'val_loss_r': [],
            'lr': [] 
        }

    def _get_device(self):
        if torch.backends.mps.is_available():
            print("Using Apple MPS acceleration.")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("Using CUDA.")
            return torch.device("cuda")
        else:
            print("Using CPU.")
            return torch.device("cpu")

    def _setup_data(self):
        print("Loading Datasets...")
        train_ds = LineModPoseDataset(self.cfg['split_train'], self.cfg['dataset_root'], mode='train')
        val_ds = LineModPoseDataset(self.cfg['split_val'], self.cfg['dataset_root'], mode='val')
        
        train_loader = DataLoader(train_ds, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=1)
        
        print(f"Data Loaded: {len(train_ds)} Train, {len(val_ds)} Val")
        return train_loader, val_loader

    def _setup_model(self):
        module = PoseResNetRGBD(pretrained=True).to(self.device)
        optimizer = optim.Adam(module.parameters(), lr=self.cfg['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.cfg['scheduler_patience']
        )
        return module, optimizer, scheduler

    def train_epoch(self, epoch):
        
        self.module.train()
        
        running_loss = 0.0
        running_loss_t = 0.0
        running_loss_r = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg['epochs']} [Train]")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            depth_images = batch['depth'].to(self.device)  
            bboxes = batch['bbox'].to(self.device)
            intrinsics = batch['cam_params'].to(self.device)
            gt_translation = batch['translation'].to(self.device)
            gt_quaternion = batch['quaternion'].to(self.device)
            class_ids = batch['class_id']

            RGBD_image = torch.cat((images, depth_images), dim=1)  # Concatenate along channel dimension

            self.optimizer.zero_grad()
            
            

            # 1. Translation (DepthNet + Pinhole)
            r_pred, t_pred = self.module(RGBD_image)  # [B, 1] or [B]
            

            
            loss_t = self.criterion_trans(t_pred, gt_translation)

            loss_r = self.criterion_rot(r_pred, gt_quaternion)

            # 3. Total Loss
            total_loss = (self.cfg['alpha'] * loss_t) + (self.cfg['beta'] * loss_r)

            # ✅ Backprop and optimize both networks
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()
            running_loss_t += loss_t.item()
            running_loss_r += loss_r.item()
            progress_bar.set_postfix({'T_loss': loss_t.item(), 'R_loss': loss_r.item()})
            n = len(self.train_loader)
        return running_loss / n, running_loss_t / n, running_loss_r / n

    def validate(self):
        # ✅ CORRECT: Set both networks to eval mode
        self.module.eval()
        
        running_val_loss = 0.0
        running_val_loss_t = 0.0
        running_val_loss_r = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                depth_images = batch['depth'].to(self.device)  # ✅ Fixed
                bboxes = batch['bbox'].to(self.device)
                intrinsics = batch['cam_params'].to(self.device)
                gt_translation = batch['translation'].to(self.device)
                gt_quats = batch['quaternion'].to(self.device)

                RGBD_image = torch.cat((images, depth_images), dim=1)  # Concatenate along channel dimension

                r_pred, t_pred = self.module(RGBD_image)  # [B, 1] or [B]
                loss_r = self.criterion_rot(r_pred, gt_quats)
                loss_t = self.criterion_trans(t_pred, gt_translation)

                total_loss = (self.cfg['alpha'] * loss_t) + (self.cfg['beta'] * loss_r)
                running_val_loss += total_loss.item()
                running_val_loss_t += loss_t.item()
                running_val_loss_r += loss_r.item()
                n = len(self.val_loader)
        return running_val_loss / n, running_val_loss_t / n, running_val_loss_r / n

    def run(self):
        print(f"Starting Training for {self.cfg['epochs']} epochs...")
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_d_net_state = None

        for epoch in range(self.cfg['epochs']):
            t_loss, t_loss_t, t_loss_r = self.train_epoch(epoch)
            v_loss, v_loss_t, v_loss_r = self.validate()
            
            self.history['train_loss'].append(t_loss)
            self.history['train_loss_t'].append(t_loss_t)
            self.history['train_loss_r'].append(t_loss_r)
            
            self.history['val_loss'].append(v_loss)
            self.history['val_loss_t'].append(v_loss_t)
            self.history['val_loss_r'].append(v_loss_r)

            print(f"Epoch {epoch+1}: Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

            self.scheduler.step(v_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            # Checkpoint & Early Stopping
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_d_net_state = copy.deepcopy(self.module.state_dict())
                
                early_stop_counter = 0
                torch.save(self.module.state_dict(), os.path.join(self.save_dir, 'best_RGBD_ResNet.pth'))
                print("🚀 New Best Model Saved!")
            else:
                early_stop_counter += 1
                print(f"⚠️ No improvement for {early_stop_counter}/{self.cfg['early_stop_patience']}")

            if early_stop_counter >= self.cfg['early_stop_patience']:
                print("\n⏹️ Early Stopping Triggered!")
                break
        
        self.plot_results()

    def plot_results(self):
        """
        Genera tre grafici separati per monitorare le loss e i momenti di LR drop.
        """
        epochs = range(len(self.history['train_loss']))
        lr_history = self.history['lr']
        
        # Identifica le epoche in cui il Learning Rate è stato ridotto
        lr_drops = [i for i in range(1, len(lr_history)) if lr_history[i] < lr_history[i-1]]

        # Setup della figura con 3 subplots orizzontali
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        
        # Configurazione dei dati da plottare
        metrics = [
            {
                'title': 'Total Weighted Loss',
                'train': self.history['train_loss'],
                'val': self.history['val_loss'],
                'color': 'tab:blue'
            },
            {
                'title': f'Translation Loss (Alpha={self.cfg["alpha"]})',
                'train': self.history['train_loss_t'],
                'val': self.history['val_loss_t'],
                'color': 'tab:orange'
            },
            {
                'title': f'Rotation Loss (Beta={self.cfg["beta"]})',
                'train': self.history['train_loss_r'],
                'val': self.history['val_loss_r'],
                'color': 'tab:green'
            }
        ]

        for i, m in enumerate(metrics):
            ax = axes[i]
            # Plot linee principali
            ax.plot(epochs, m['train'], label='Train', color=m['color'], alpha=0.4, linestyle='--')
            ax.plot(epochs, m['val'], label='Validation', color=m['color'], linewidth=2)
            
            # Aggiunta linee verticali per ogni LR Drop
            for drop_idx in lr_drops:
                ax.axvline(x=drop_idx, color='red', linestyle=':', linewidth=1.5, 
                           label='LR Drop' if drop_idx == lr_drops[0] else "")
            
            # Estetica e labels
            ax.set_title(m['title'], fontsize=14, fontweight='bold')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss Value')
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc='upper right')

        plt.suptitle("6D Pose Estimation Training Analysis", fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Salvataggio e visualizzazione
        plot_path = os.path.join(self.save_dir, "training_metrics_split.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grafici salvati correttamente in: {plot_path}")
        plt.show()