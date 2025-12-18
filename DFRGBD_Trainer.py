import torch
import torch.optim as optim
import os
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from models.DenseFusion_RGBD_Net import DenseFusion_RGBD_Net
from utils.Posenet_utils.attention import GeometricAttention
from utils.Posenet_utils.quaternion_Loss import QuaternionLoss

class DenseFusion_RGBD_Trainer:
    def __init__(self, config):
        """
        Inizializza il trainer con un dizionario di configurazione.
        """
        self.cfg = config
        self.device = self._get_device()
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_loader, self.val_loader = self._setup_data()
        self.model, self.optimizer, self.scheduler = self._setup_model()
        self.criterion_trans = torch.nn.L1Loss()
        self.criterion_rot = QuaternionLoss()
        
        # Metrics
        self.history = {'train_loss': [], 'val_loss': []}

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
        model = DenseFusion_RGBD_Net(pretrained=True).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.cfg['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.cfg['scheduler_patience']
        )
        return model, optimizer, scheduler

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg['epochs']} [Train]")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            depths = batch['depth'].to(self.device)
            bboxes = batch['bbox'].to(self.device)
            #intrinsics = batch['cam_params'].to(self.device)
            gt_translation = batch['translation'].to(self.device)
            gt_quaternion = batch['quaternion'].to(self.device)
            class_ids = batch['class_id']

            self.optimizer.zero_grad()

            pred_rot, pred_trans = self.model(images,depths)

            loss_t = self.criterion_trans(pred_trans, gt_translation)
            loss_r = self.criterion_rot(pred_rot, gt_quaternion)

            total_loss = (self.cfg['alpha'] * loss_t) + (self.cfg['beta'] * loss_r)

            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()
            progress_bar.set_postfix({'T_loss': loss_t.item(), 'R_loss': loss_r.item()})
            
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                depths = batch['depth'].to(self.device)
                bboxes = batch['bbox'].to(self.device)
                intrinsics = batch['cam_params'].to(self.device)
                gt_translation = batch['translation'].to(self.device)
                gt_quaternion = batch['quaternion'].to(self.device)
                class_ids = batch['class_id']

                pred_rot, pred_trans = self.model(images,depths)

                loss_t = self.criterion_trans(pred_trans, gt_translation)
                loss_r = self.criterion_rot(pred_rot, gt_quaternion )                

                total_loss = (self.cfg['alpha'] * loss_t) + (self.cfg['beta'] * loss_r)
                running_val_loss += total_loss.item()
                
        return running_val_loss / len(self.val_loader)

    def run(self):
        print(f"Starting Training for {self.cfg['epochs']} epochs...")
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None

        for epoch in range(self.cfg['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            self.scheduler.step(val_loss)

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best__DFRGBD.pth'))
                print("----> New Best Model Saved! <----")
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter}/{self.cfg['early_stop_patience']}")

            if early_stop_counter >= self.cfg['early_stop_patience']:
                print("\n Early Stopping Triggered!")
                break
        
        self.plot_results()

    def plot_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f"Training Results (LR={self.cfg['lr']})")
        plt.savefig(os.path.join(self.save_dir, "training_curve.png"))
        plt.show()