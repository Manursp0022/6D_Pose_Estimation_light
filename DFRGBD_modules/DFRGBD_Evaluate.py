import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

# Import dei tuoi modelli e utils
from models.DenseFusion_RGBD_Net import DenseFusion_RGBD_Net 
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.PoseEvaluator import PoseEvaluator 
from utils.Posenet_utils.utils_geometric import crop_square_resize

class DF_RGBD_Net_Evaluator:
    def __init__(self, config):
        self.cfg = config
        self.device = self._get_device()
        print(f"Initializing Evaluator on: {self.device}")

        # Diametri oggetti (mm)
        self.DRS = {
            1: 102.09, 2: 247.50, 4: 172.49, 5: 201.40, 6: 154.54, 8: 261.47,
            9: 108.99, 10: 164.62, 11: 175.88, 12: 145.54, 13: 278.07, 14: 282.60, 15: 212.35
        }
        
        # Nomi oggetti
        self.OBJ_NAMES = {
            1: "Ape", 2: "Benchvise", 4: "Cam", 5: "Can", 6: "Cat", 8: "Driller",
            9: "Duck", 10: "Eggbox", 11: "Glue", 12: "Holepuncher", 13: "Iron", 14: "Lamp", 15: "Phone"
        }

        self.val_loader = self._setup_data()
        self.models_3d = self._load_3d_models()
        self.DF_RGBD = self._setup_DF_RGBD()
        self.YOLO = self._setup_YOLO()
        
        self.metric_calculator = PoseEvaluator(np.eye(3))

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
        print(" Loading Validation Dataset...")
        val_ds = LineModPoseDataset(self.cfg['split_val'], self.cfg['dataset_root'], mode='val')
        # num_workers=0 per evitare crash su Mac
        return DataLoader(val_ds, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=2)

    def _setup_DF_RGBD(self):
        print(" Loading DenseFusion RGB-D Model...")
        model = DenseFusion_RGBD_Net(pretrained=False).to(self.device)
        weights_path = os.path.join(self.cfg['model_dir'], 'best__DFRGBD.pth')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f" Weights not found at: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        return model

    def _setup_YOLO(self):
        print(" Loading YOLO model....")
        weights_path = os.path.join(self.cfg['model_dir'], 'best_YOLO.pt')
        return YOLO(weights_path).to(self.device)

    def _load_3d_models(self):
        print(" Loading 3D Models (.ply)...")
        models_3d = {}
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        for obj_id in self.DRS.keys():
            path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
            if os.path.exists(path):
                ply = PlyData.read(path)
                vertex = ply['vertex']
                pts_mm = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
                models_3d[obj_id] = pts_mm / 1000.0 
        return models_3d

    def _quaternion_to_matrix(self, quats):
        if isinstance(quats, torch.Tensor): quats = quats.cpu().numpy()
        quats_scipy = np.concatenate([quats[:, 1:], quats[:, 0:1]], axis=1)
        return R.from_quat(quats_scipy).as_matrix()

    def run(self):
        print("\n Starting Evaluation (Pure RGB-D Output)...")
        lm_to_yolo = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12}
        
        resnet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        
        class_stats = {oid: {'correct': 0, 'total': 0, 'errors': []} for oid in self.DRS.keys()}
        total_correct = 0
        total_preds = 0
        all_errors = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                paths = batch['path']
                gt_translation = batch['translation'].to(self.device)
                gt_quats = batch['quaternion'].to(self.device)
                class_ids = batch['class_id'].numpy()
                rgb_batch = batch['image'].to(self.device)
                depth_batch = batch['depth'].to(self.device)
                """
                # YOLO Inference
                yolo_results = self.YOLO(paths, conf=0.5, verbose=False)

                resnet_rgb_list = []
                resnet_depth_list = []
                valid_indices = []

                for i, result in enumerate(yolo_results):
                    target_id = int(class_ids[i])
                    target_yolo = lm_to_yolo.get(target_id, -1)
                    
                    found_box = None
                    best_conf = -1
                    for box in result.boxes:
                        if int(box.cls[0]) == target_yolo and float(box.conf[0]) > best_conf:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            found_box = torch.tensor([x1, y1, x2-x1, y2-y1])
                            best_conf = box.conf[0]
                    
                    if found_box is not None:
                        img_raw = cv2.imread(paths[i])
                        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                        
                        d_path = paths[i].replace('rgb', 'depth')
                        d_raw = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                        
                        rgb_crop = crop_square_resize(img_raw, found_box, 224, is_depth=False)
                        d_crop = crop_square_resize(d_raw, found_box, 224, is_depth=True)
                        
                        resnet_rgb_list.append(resnet_transform(rgb_crop))
                        resnet_depth_list.append(torch.from_numpy(d_crop).float().unsqueeze(0))
                        
                        valid_indices.append(i)

                if len(valid_indices) == 0: continue
                """

                #rgb_batch = torch.stack(resnet_rgb_list).to(self.device)
                #depth_batch = torch.stack(resnet_depth_list).to(self.device)
                
                # 2. Network Inference (Coarse Pose)
                pred_quats, pred_trans = self.DF_RGBD(rgb_batch, depth_batch)
                
                #pred_R_np = self._quaternion_to_matrix(pred_quats)
                #pred_t_np = pred_trans.cpu().numpy()
                
                #gt_trans_np = gt_translation[valid_indices].cpu().numpy()
                #gt_R_np = self._quaternion_to_matrix(gt_quats[valid_indices])

                pred_R_np = self._quaternion_to_matrix(pred_quats)
                pred_t_np = pred_trans.cpu().numpy()
                
                gt_trans_np = gt_translation.cpu().numpy()
                gt_R_np = self._quaternion_to_matrix(gt_quats)

                batch_size = rgb_batch.shape[0]
                
                """
                # 3. Metric Calculation
                for k, original_idx in enumerate(valid_indices):
                    obj_id = int(class_ids[original_idx])
                    if obj_id not in self.models_3d: continue

                    pts_3d = self.models_3d[obj_id]
                    add_error = self.metric_calculator.calculate_metric(
                        pred_R_np[k], pred_t_np[k], gt_R_np[k], gt_trans_np[k], pts_3d, obj_id
                    )
                    
                    diameter = self.DRS[obj_id] / 1000.0
                    is_correct = self.metric_calculator.is_pose_correct(add_error, 0.1 * diameter)
                    
                    if is_correct: total_correct += 1
                    total_preds += 1
                    all_errors.append(add_error * 100.0)

                    if obj_id in class_stats:
                        class_stats[obj_id]['total'] += 1
                        class_stats[obj_id]['errors'].append(add_error * 100.0)
                        if is_correct: class_stats[obj_id]['correct'] += 1
                """
                for k in range(batch_size):
                    obj_id = int(class_ids[k])
                    
                    # Salta se non abbiamo il modello 3D (non dovrebbe succedere se il dataset è ok)
                    if obj_id not in self.models_3d: 
                        continue

                    pts_3d = self.models_3d[obj_id]
                    
                    # Calcolo ADD
                    add_error = self.metric_calculator.calculate_metric(
                        pred_R_np[k], pred_t_np[k], 
                        gt_R_np[k], gt_trans_np[k], 
                        pts_3d, obj_id
                    )
                    
                    diameter = self.DRS[obj_id] / 1000.0
                    is_correct = self.metric_calculator.is_pose_correct(add_error, 0.1 * diameter)
                    
                    # Aggiorna Statistiche
                    if is_correct: total_correct += 1
                    total_preds += 1
                    all_errors.append(add_error * 100.0) # cm

                    if obj_id in class_stats:
                        class_stats[obj_id]['total'] += 1
                        class_stats[obj_id]['errors'].append(add_error * 100.0)
                        if is_correct: class_stats[obj_id]['correct'] += 1

        accuracy = (total_correct / total_preds * 100.0) if total_preds > 0 else 0.0
        mean_add = np.mean(all_errors) if len(all_errors) > 0 else 0.0
        
        self._print_report(accuracy, mean_add, total_preds)
        self._plot_per_class_results(class_stats)
    
    def _print_report(self, accuracy, mean_add, total):
        print("\n" + "="*60)
        print("FINAL REPORT (ADD-0.1d Metric)")
        print("="*60)
        print(f" Samples Eval:   {total}")
        print("-" * 40)
        print(f" Total Accuracy: {accuracy:.2f} %")
        print(f" Mean ADD Error: {mean_add:.2f} cm")
        print("="*60)

    def _plot_per_class_results(self, class_stats):
        print(" Generating Plots...")
        obj_ids = sorted(list(class_stats.keys()))
        labels = [self.OBJ_NAMES.get(oid, str(oid)) for oid in obj_ids]
        
        accuracies = []
        mean_errors = []
        thresholds_cm = []
        
        for oid in obj_ids:
            stats = class_stats[oid]
            total = stats['total']
            if total > 0:
                acc = (stats['correct'] / total) * 100.0
                avg_err = np.mean(stats['errors'])
            else:
                acc = 0.0
                avg_err = 0.0
            accuracies.append(acc)
            mean_errors.append(avg_err)
            thresholds_cm.append((self.DRS[oid] / 1000.0 * 0.1) * 100.0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        bars = ax1.bar(labels, accuracies, color='skyblue', edgecolor='black')
        ax1.set_ylabel('Accuracy (ADD-0.1d %)')
        ax1.set_title('Per-Class Accuracy')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        x = np.arange(len(labels))
        width = 0.35
        ax2.bar(x - width/2, mean_errors, width, label='Mean ADD Error (cm)', color='salmon')
        ax2.bar(x + width/2, thresholds_cm, width, label='Threshold (0.1*D)', color='lightgreen', alpha=0.7)
        ax2.set_ylabel('Distance (cm)')
        ax2.set_title('Mean ADD Error vs Threshold (10% Diam)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(self.cfg['save_dir'], 'final_evaluation_rgbd.png')
        plt.savefig(save_path)
        print(f" Plot saved to: {save_path}")
        plt.show()