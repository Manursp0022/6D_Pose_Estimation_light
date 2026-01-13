import torch
import torch.optim as optim
import os
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.utils_geometric import solve_pinhole_diameter, backproject_bbox_to_3d
from Refinement_Section.Pinhole_Refinement_Z import ImageBasedTranslationNet as TinyPinholeRefiner
from models.Posenet import PoseResNet
from utils.Posenet_utils.quaternion_Loss import QuaternionLoss
from utils.Posenet_utils.translation_Loss import weightedTranslationLoss

class PoseNetTrainerRefined:
    def __init__(self, config):
        """
        Inizializza il trainer con un dizionario di configurazione.
        """
        self.cfg = config
        self.device = self._get_device()
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.DRS = {
            1: 102.09865663, 2: 247.50624233, 4: 172.49224865,
            5: 201.40358597, 6: 154.54551808, 8: 261.47178102,
            9: 108.99920102, 10: 164.62758848, 11: 175.88933422,
            12: 145.54287471, 13: 278.07811733, 14: 282.60129399,
            15: 212.35825148
        }

        self.train_loader, self.val_loader = self._setup_data()
        self.model, self.refiner, self.optimizer, self.scheduler = self._setup_model()
        self.criterion_trans = weightedTranslationLoss()
        self.criterion_rot = QuaternionLoss()
        
        # Metrics
        self.history = {'train_loss': [], 'val_loss': [], 'resnet_lr': [], 'refiner_lr': []}

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
        val_loader = DataLoader(val_ds, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=2)
        
        print(f"Data Loaded: {len(train_ds)} Train, {len(val_ds)} Val")
        return train_loader, val_loader

    def _setup_model(self):
        model = PoseResNet(pretrained=True).to(self.device)
        refiner = TinyPinholeRefiner().to(self.device)
        optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': self.cfg['lr']},
        {'params': refiner.parameters(), 'lr': self.cfg['lr'] * 0.1}  # 10x slower
        ])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.cfg['scheduler_patience']
        )
        return model, refiner, optimizer, scheduler

    def _get_diameters_tensor(self, class_ids):
        """Helper per creare il tensore dei diametri al volo"""
        current_diameters = []
        for cid in class_ids:
            # Recupera diametro in mm e converti in METRI (/1000.0)
            d_meters = self.DRS[cid.item()] / 1000.0
            current_diameters.append(d_meters)
        return torch.tensor(current_diameters, dtype=torch.float32).to(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg['epochs']} [Train]")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            #bboxes = batch['bbox_norm'].to(self.device)
            bboxes_raw = batch['gt_bbox'].to(self.device)
            intrinsics = batch['cam_params_raw'].to(self.device)
            gt_translation = batch['translation'].to(self.device)
            gt_quaternion = batch['quaternion'].to(self.device)
            class_ids = batch['class_id'].to(self.device)

            self.optimizer.zero_grad()
            
            # Translation (Refined Pinhole - Yes Gradient)
            diameters = self._get_diameters_tensor(class_ids)

            z_pred = self.refiner(images, bboxes_raw)

            fx, fy, cx, cy = [intrinsics[:, i].unsqueeze(1) for i in [0, 1, 2, 3]]
            
            #print("Intrinsics fx, fy, cx, cy:", fx, fy, cx, cy)
            pred_trans = backproject_bbox_to_3d(bboxes_raw, z_pred, fx, fy, cx, cy)
            """print("Predicted Translation:", pred_trans)
            print("Ground Truth Translation:", gt_translation)"""
            loss_t = self.criterion_trans(pred_trans, gt_translation, weight=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(self.device))

            #  Rotation (ResNet - Yes Gradient)
            pred_quats = self.model(images)
            loss_r = self.criterion_rot(pred_quats, gt_quaternion)

            # 3. Total Loss
            total_loss = (self.cfg['alpha'] * loss_t) + (self.cfg['beta'] * loss_r)

            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()
            progress_bar.set_postfix({'T_loss': loss_t.item(), 'R_loss': loss_r.item()})
        
        #print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} (T: {loss_t.item():.4f}, R: {loss_r.item():.4f}) | Val Loss: {val_loss:.4f}")
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        self.refiner.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                bboxes = batch['bbox_norm'].to(self.device)
                bboxes_raw = batch['gt_bbox'].to(self.device)
                intrinsics = batch['cam_params_raw'].to(self.device)
                gt_translation = batch['translation'].to(self.device)
                gt_quats = batch['quaternion'].to(self.device)
                class_ids = batch['class_id'].to(self.device)

                # Translation
                
                fx, fy, cx, cy = [intrinsics[:, i].unsqueeze(1) for i in [0, 1, 2, 3]]
                #print(fx, fy, cx, cy)
                #print("Intrinsics fx, fy, cx, cy:", fx, fy, cx, cy)
                z_pred = self.refiner(images, bboxes_raw)

                pred_trans = backproject_bbox_to_3d(bboxes_raw, z_pred, fx, fy, cx, cy)
                #print("Refined Translation:", pred_translation)
                #print("Ground Truth Translation:", gt_translation)
                loss_t = self.criterion_trans(pred_trans, gt_translation, weight=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(self.device))

                # Rotation
                pred_quats = self.model(images)
                loss_r = self.criterion_rot(pred_quats, gt_quats)

                total_loss = (self.cfg['alpha'] * loss_t) + (self.cfg['beta'] * loss_r)
                
                running_val_loss += total_loss.item()
                
        return running_val_loss / len(self.val_loader)

    def run(self):
        print(f"Starting Training for {self.cfg['epochs']} epochs...")
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        resnet_lr = self.cfg['lr']
        refiner_lr = self.cfg['lr'] * 0.1
        for epoch in range(self.cfg['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['resnet_lr'].append(resnet_lr) 
            self.history['refiner_lr'].append(refiner_lr)

            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            self.scheduler.step(val_loss)
            resnet_lr = self.optimizer.param_groups[0]['lr']
            refiner_lr = self.optimizer.param_groups[1]['lr']
            
            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_posenet_baseline_w_refiner.pth'))
                torch.save(self.refiner.state_dict(), os.path.join(self.save_dir, 'best_pinhole_refiner_z.pth'))
                print("----> New Best Model Saved! <----")
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter}/{self.cfg['early_stop_patience']}")

            if early_stop_counter >= self.cfg['early_stop_patience']:
                print("\n Early Stopping Triggered!")
                break
        
        self.plot_results()

    def plot_results(self):
        """
        Genera due grafici separati: uno per le Loss e uno per le Learning Rates.
        """
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Creiamo una figura con 2 subplot (2 righe, 1 colonna)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # --- PLOT 1: LOSS ---
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss Value')
        ax1.set_title(f"Training & Validation Loss\n(Alpha={self.cfg['alpha']}, Beta={self.cfg['beta']})")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- PLOT 2: LEARNING RATES ---
        # Usiamo una scala logaritmica se i tassi variano di molto
        ax2.plot(epochs, self.history['resnet_lr'], 'g--', label='ResNet LR')
        ax2.plot(epochs, self.history['refiner_lr'], 'm--', label='Refiner LR')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')  # Opzionale: utile se scendono di diversi ordini di grandezza
        ax2.set_title("Learning Rate Schedule")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        
        # Salvataggio
        save_path = os.path.join(self.save_dir, "training_metrics.png")
        plt.savefig(save_path)
        print(f"📈 Grafici salvati in: {save_path}")
        plt.show()
if __name__ == "__main__":
    config = {
        'split_train': 'data/autosplit_train_ALL.txt',
        'split_val': 'data/autosplit_val_ALL.txt',
        'dataset_root': 'dataset/Linemod_preprocessed',
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 50,
        'alpha': 0.1,
        'beta': 1.0,
        'scheduler_patience': 5,
        'early_stop_patience': 10,
        'save_dir': 'checkpoints/'
    }
    
    trainer = PoseNetTrainerRefined(config)
    trainer.run()