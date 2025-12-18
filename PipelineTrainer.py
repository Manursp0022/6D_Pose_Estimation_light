import torch
import torch.optim as optim
import os
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import pynvml
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from models.RGB_D_ResNet import PoseResNetRGBD
from utils.Posenet_utils.quaternion_Loss import QuaternionLoss
from utils.Posenet_utils.utils_geometric import crop_square_resize, image_transformation
import ultralytics
import cv2
import torchvision.transforms as transforms
import random

class GPUTracker:
    """Helper class to monitor NVIDIA GPU metrics."""
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
        except Exception as e:
            print(f"⚠️ GPU tracking unavailable: {e}")
            self.enabled = False

    def get_power(self):
        """Returns power usage in Watts."""
        if not self.enabled: return 0.0
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
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
        # Image size used for cropping/resizing (falls back to 224)
        self.img_size = config.get('img_size', 224)
        os.makedirs(self.save_dir, exist_ok=True)

        self.gpu_tracker = GPUTracker()

        self.yolo_to_folder = {
        0: '01', 1: '02', 2: '04', 3: '05', 4: '06', 5: '08',
        6: '09', 7: '10', 8: '11', 9: '12', 10: '13', 11: '14', 12: '15'
        }

        self.train_loader, self.val_loader = self._setup_data()
        self.module, self.optimizer, self.scheduler = self._setup_model()
        self.criterion_trans = torch.nn.L1Loss()
        self.criterion_rot = QuaternionLoss()
        self.yolo = self._setup_yolo()
        
        # Data augmentation for training only
        self.rgb_augmentation = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])
        
        # Metrics
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_loss_t': [], 'val_loss_t': [],
            'train_loss_r': [], 'val_loss_r': [],
            'lr': [],
            'train_time': [], 'val_time': [],
            'avg_power_w': [], 'throughput': [] # Samples per second
        }
        
    def apply_bbox_jitter(self, bbox_center, jitter_range=0.05):
        """
        Apply random jitter to bounding box center for training augmentation.
        
        Args:
            bbox_center: Tensor of shape [B, 2] with normalized bbox centers
            jitter_range: Maximum jitter as fraction of image size (default: 0.05 = 5%)
        
        Returns:
            Jittered bbox_center
        """
        jitter = torch.randn_like(bbox_center) * jitter_range
        jittered_center = bbox_center + jitter
        # Clamp to valid range [0, 1]
        jittered_center = torch.clamp(jittered_center, 0.0, 1.0)
        return jittered_center
    
    def _setup_yolo(self):
        # Carica il modello YOLOv8 pre-addestrato
        yolo_model = ultralytics.YOLO(self.cfg['yolo_weights']).to(self.device)
        return yolo_model
        
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
        optimizer = optim.Adam(module.parameters(), lr=self.cfg['lr'], weight_decay=1e-6)
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
        start_time = time.time()
        power_readings = []
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            depth_images = batch['depth'].to(self.device)  
            bboxes = batch['bbox'].to(self.device)
            gt_translation = batch['translation'].to(self.device)
            gt_quaternion = batch['quaternion'].to(self.device)
            

            RGBD_image = torch.cat((images, depth_images), dim=1)  # Concatenate along channel dimension

            power_readings.append(self.gpu_tracker.get_power())
            self.optimizer.zero_grad()

            #bboxes are in [x, y, w, h] format 
            bx = int(bboxes[:, 0] + bboxes[:, 2] / 2.0)  # center x
            by = int(bboxes[:, 1] + bboxes[:, 3] / 2.0)  # center y

            bx_norm = bx / 640.0 
            by_norm = by / 480.0
            bbox_center = torch.stack([bx_norm, by_norm], dim=1)

            # 1. Translation (DepthNet + Pinhole)
            r_pred, t_pred = self.module(RGBD_image, bbox_center)  # [B, 1] or [B]
            

            
            loss_t = self.criterion_trans(t_pred, gt_translation/1000.0)  # Convert mm to meters

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

        epoch_duration = time.time() - start_time
        avg_p = sum(power_readings)/len(power_readings) if power_readings else 0
        samples_sec = len(self.train_loader.dataset) / epoch_duration

        self.history['train_time'].append(epoch_duration)
        self.history['avg_power_w'].append(avg_p)
        self.history['throughput'].append(samples_sec)

        n = len(self.train_loader)
        return running_loss / n, running_loss_t / n, running_loss_r / n

    def validate(self):
        # ✅ CORRECT: Set both networks to eval mode
        self.module.eval()
        self.yolo.eval()

        start_time = time.time()

        running_val_loss = 0.0
        running_val_loss_t = 0.0
        running_val_loss_r = 0.0
        valid_samples = 0
        skipped_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                
                path = batch['path']
                
                gt_translation = batch['translation'].to(self.device)
                gt_class = batch['class_id'].to(self.device)
                gt_quats = batch['quaternion'].to(self.device)

                yolo_outputs = self.yolo(path, verbose=False)  # Assuming images are in the correct format for YOLO

                batch_size = gt_translation.shape[0]
                for i in range(batch_size):
                    # Get YOLO detections for this image
                    detections = yolo_outputs[i].boxes

                    # Check if YOLO detected any objects
                    if len(detections) == 0:
                        skipped_samples += 1
                        continue

                    # Require the detection to match the ground-truth class.
                    # Build list of candidate detection indices whose mapped class == gt_class
                    gt_c = None
                    if gt_class is not None:
                        gt_c = gt_class[i].item() if hasattr(gt_class[i], 'item') else int(gt_class[i])

                    candidates = []
                    for idx in range(len(detections)):
                        try:
                            cls_idx = int(detections.cls[idx].item()) if hasattr(detections.cls[idx], 'item') else int(detections.cls[idx])
                        except Exception:
                            cls_idx = int(detections.cls[idx])

                        mapped_folder = self.yolo_to_folder.get(cls_idx, None)
                        mapped_int = int(mapped_folder) if mapped_folder is not None else cls_idx

                        if gt_c is not None and mapped_int == int(gt_c):
                            candidates.append(idx)

                    # If no detections match the ground truth class, skip sample
                    if len(candidates) == 0:
                        skipped_samples += 1
                        # Debugging info for mismatches
                        if gt_c is not None:
                            print(f"YOLO: no detection matching gt class {int(gt_c)}; skipping sample")
                        continue

                    # From candidate detections, pick the one with highest confidence
                    confidences = detections.conf
                    best_idx = candidates[0]
                    best_conf = confidences[best_idx]
                    for c in candidates[1:]:
                        if confidences[c] > best_conf:
                            best_idx = c
                            best_conf = confidences[c]

                    # Use the GT class as predicted_class (we selected matching detection)
                    
                    predicted_bbox = detections.xywh[best_idx]
                    
                    # Calculate bbox center from YOLO prediction
                    bx = predicted_bbox[0]
                    by = predicted_bbox[1] 
                    w, h = predicted_bbox[2], predicted_bbox[3]
                    # Normalize bbox center (NO JITTER during validation)
                    bx_norm = bx / 640.0
                    by_norm = by / 480.0
                    bbox_center = torch.stack([bx_norm, by_norm], dim=0).unsqueeze(0)  # [1, 2]
                    
                    img = cv2.imread(path[i])
                    if img is None:
                        print(f"Warning: Could not load image {path[i]}")
                        skipped_samples += 1
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    

                    depth_path = path[i].replace('rgb', 'depth')
                    d_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    
                    if d_img is None:
                        print(f"Warning: Could not load depth image for {depth_path[i]}")
                        skipped_samples += 1
                        continue
                    
                    gt_x = bx-(w / 2.0)
                    gt_y = by-(h / 2.0)
                    predicted_bbox = [gt_x, gt_y, w, h]
                    # Crop and resize based on YOLO bbox (xywh -> xyxy conversion done above)
                    img_cropped = crop_square_resize(img, predicted_bbox, self.img_size, is_depth=False)
                    d_img_cropped = crop_square_resize(d_img, predicted_bbox, self.img_size, is_depth=True)
                    
                    # Convert RGB to tensor and normalize (NO AUGMENTATION during validation)
                    img_tensor = image_transformation(img_cropped)  # Apply ImageNet normalization
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)  # [3, H, W]
                    
                    # Convert depth to tensor (no normalization)
                    depth_tensor = torch.from_numpy(d_img_cropped).float().unsqueeze(0)
                    
                    # Concatenate RGB and Depth
                    RGBD_image = torch.cat((img_tensor, depth_tensor), dim=1)  # [1, 4, H, W]
                    
                    # Forward pass through pose network
                    r_pred, t_pred = self.module(RGBD_image, bbox_center)
                    
                    # Calculate losses
                    loss_r = self.criterion_rot(r_pred, gt_quats[i:i+1])
                    loss_t = self.criterion_trans(t_pred, gt_translation[i:i+1] / 1000.0)
                    
                    total_loss = (self.cfg['alpha'] * loss_t) + (self.cfg['beta'] * loss_r)
                    
                    running_val_loss += total_loss.item()
                    running_val_loss_t += loss_t.item()
                    running_val_loss_r += loss_r.item()
                    valid_samples += 1
        
        
        val_duration = time.time() - start_time
        self.history['val_time'].append(val_duration)
        # Print statistics
        total_samples = valid_samples + skipped_samples
        if total_samples > 0:
            print(f"\nValidation Stats: {valid_samples}/{total_samples} samples used "
                  f"({100*valid_samples/total_samples:.1f}%), {skipped_samples} skipped")
        
        # Avoid division by zero
        if valid_samples == 0:
            print("⚠️ Warning: No valid samples detected by YOLO during validation!")
            return float('inf'), float('inf'), float('inf')

        
        return running_val_loss / valid_samples, running_val_loss_t / valid_samples, running_val_loss_r / valid_samples

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
        Generates comprehensive plots for all training metrics including losses, learning rate, 
        training times, power consumption, and throughput.
        """

        epochs = range(len(self.history['train_loss']))
        lr_history = self.history['lr']
        
        # Identify epochs where Learning Rate was reduced
        lr_drops = [i for i in range(1, len(lr_history)) if lr_history[i] < lr_history[i-1]]

        # Setup figure with 2 rows and 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(21, 12))
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # ========== ROW 1: Loss Metrics ==========
        
        # Plot 1: Total Weighted Loss
        ax = axes[0]
        ax.plot(epochs, self.history['train_loss'], label='Train', color='tab:blue', alpha=0.4, linestyle='--')
        ax.plot(epochs, self.history['val_loss'], label='Validation', color='tab:blue', linewidth=2)
        for drop_idx in lr_drops:
            ax.axvline(x=drop_idx, color='red', linestyle=':', linewidth=1.5, 
                    label='LR Drop' if drop_idx == lr_drops[0] else "")
        ax.set_title('Total Weighted Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss Value')
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc='upper right')
        
        # Plot 2: Translation Loss
        ax = axes[1]
        ax.plot(epochs, self.history['train_loss_t'], label='Train', color='tab:orange', alpha=0.4, linestyle='--')
        ax.plot(epochs, self.history['val_loss_t'], label='Validation', color='tab:orange', linewidth=2)
        for drop_idx in lr_drops:
            ax.axvline(x=drop_idx, color='red', linestyle=':', linewidth=1.5, 
                    label='LR Drop' if drop_idx == lr_drops[0] else "")
        ax.set_title(f'Translation Loss (Alpha={self.cfg["alpha"]})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss Value')
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc='upper right')
        
        # Plot 3: Rotation Loss
        ax = axes[2]
        ax.plot(epochs, self.history['train_loss_r'], label='Train', color='tab:green', alpha=0.4, linestyle='--')
        ax.plot(epochs, self.history['val_loss_r'], label='Validation', color='tab:green', linewidth=2)
        for drop_idx in lr_drops:
            ax.axvline(x=drop_idx, color='red', linestyle=':', linewidth=1.5, 
                    label='LR Drop' if drop_idx == lr_drops[0] else "")
        ax.set_title(f'Rotation Loss (Beta={self.cfg["beta"]})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss Value')
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc='upper right')
        
        # ========== ROW 2: Performance Metrics ==========
        
        # Plot 4: Learning Rate
        ax = axes[3]
        ax.plot(epochs, lr_history, label='Learning Rate', color='tab:purple', linewidth=2)
        for drop_idx in lr_drops:
            ax.axvline(x=drop_idx, color='red', linestyle=':', linewidth=1.5, 
                    label='LR Drop' if drop_idx == lr_drops[0] else "")
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')  # Log scale for better visualization
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc='upper right')
        
        # Plot 5: Training and Validation Time
        ax = axes[4]
        if self.history['train_time'] and self.history['val_time']:
            ax.plot(epochs, self.history['train_time'], label='Train Time', color='tab:cyan', linewidth=2, marker='o', markersize=3)
            ax.plot(epochs, self.history['val_time'], label='Val Time', color='tab:pink', linewidth=2, marker='s', markersize=3)
            ax.set_title('Epoch Duration', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Time (seconds)')
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No timing data available', ha='center', va='center', fontsize=12)
            ax.set_title('Epoch Duration', fontsize=14, fontweight='bold')
        
        # Plot 6: Power and Throughput (dual y-axis)
        ax = axes[5]
        if self.history['avg_power_w'] or self.history['throughput']:
            ax_power = ax
            ax_throughput = ax.twinx()
            
            if self.history['avg_power_w']:
                line1 = ax_power.plot(epochs, self.history['avg_power_w'], label='Avg Power (W)', 
                                    color='tab:red', linewidth=2, marker='o', markersize=3)
                ax_power.set_ylabel('Power (W)', color='tab:red')
                ax_power.tick_params(axis='y', labelcolor='tab:red')
            
            if self.history['throughput']:
                line2 = ax_throughput.plot(epochs, self.history['throughput'], label='Throughput (samples/s)', 
                                        color='tab:olive', linewidth=2, marker='s', markersize=3)
                ax_throughput.set_ylabel('Throughput (samples/s)', color='tab:olive')
                ax_throughput.tick_params(axis='y', labelcolor='tab:olive')
            
            ax_power.set_title('Power Consumption & Throughput', fontsize=14, fontweight='bold')
            ax_power.set_xlabel('Epochs')
            ax_power.grid(True, which="both", alpha=0.3)
            
            # Combine legends
            lines = []
            labels = []
            if self.history['avg_power_w']:
                lines.extend(line1)
                labels.append('Avg Power (W)')
            if self.history['throughput']:
                lines.extend(line2)
                labels.append('Throughput (samples/s)')
            ax_power.legend(lines, labels, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No power/throughput data available', ha='center', va='center', fontsize=12)
            ax.set_title('Power Consumption & Throughput', fontsize=14, fontweight='bold')
        
        plt.suptitle("6D Pose Estimation Training Analysis - Complete Metrics", fontsize=16, y=0.995)
        plt.tight_layout()
        
        # Save and display
        plot_path = os.path.join(self.save_dir, "training_metrics_complete.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Complete metrics plot saved to: {plot_path}")
        plt.show()


if __name__ == "__main__":
    # Example configuration
    config = {
        'dataset_root': 'C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\Linemod_preprocessed',
        'split_train': 'C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\Linemod_preprocessed\\autosplit_train_ALL.txt',
        'split_val': 'C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\dataset\\Linemod_preprocessed\\autosplit_val_ALL.txt',
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 5,
        'alpha': 1.0,  # Weight for translation loss
        'beta': 1.0,   # Weight for rotation loss
        'scheduler_patience': 5,
        'early_stop_patience': 10,
        'save_dir': './checkpoints',
        'yolo_weights': 'C:\\Users\\gabri\\Desktop\\AML project\\6D_Pose_Estimation_light\\checkpoints\\best_YOLO.pt'  # Path to YOLOv8 weights
    }

    trainer = PipelineTrainer(config)
    trainer.run()