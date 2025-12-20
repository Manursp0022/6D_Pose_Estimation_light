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

from models.DenseFusion_RGBD_Net import DenseFusion_RGBD_Net 
from models.PoseRefine_Net import PoseRefineNet
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.PoseEvaluator import PoseEvaluator 

class RefinedEvaluator:
    def __init__(self, config):
        self.cfg = config
        self.device = self._get_device()
        print(f"Initializing Refined Evaluator on: {self.device}")

        self.DRS = {
            1: 102.09, 2: 247.50, 4: 172.49, 5: 201.40, 6: 154.54, 8: 261.47,
            9: 108.99, 10: 164.62, 11: 175.88, 12: 145.54, 13: 278.07, 14: 282.60, 15: 212.35
        }
        
        self.OBJ_NAMES = {
            1: "Ape", 2: "Benchvise", 4: "Cam", 5: "Can", 6: "Cat", 8: "Driller",
            9: "Duck", 10: "Eggbox", 11: "Glue", 12: "Holepuncher", 13: "Iron", 14: "Lamp", 15: "Phone"
        }

        self.val_loader = self._setup_data()
        self.models_3d_gpu = self._load_3d_models_gpu() # Per il Refiner
        self.models_3d_cpu = self._load_3d_models_cpu() # Per calcolo metriche ADD
        
        self.model_main = self._setup_main_model()
        self.refiner = self._setup_refiner()
        
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
        return DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    def _setup_main_model(self):
        print(" Loading DenseFusion (Coarse)...")
        model = DenseFusion_RGBD_Net(pretrained=False).to(self.device)
        model.load_state_dict(torch.load(os.path.join(self.cfg['save_dir'], 'best__DFRGBD.pth'), map_location=self.device))
        model.eval()
        return model

    def _setup_refiner(self):
        print(" Loading RefineNet...")
        model = PoseRefineNet(num_points=self.cfg['num_points_mesh']).to(self.device)
        weights_path = os.path.join(self.cfg['save_dir'], 'best_refiner.pth')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Refiner weights not found at {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        return model

    def _load_3d_models_gpu(self):
        # Carica i punti per il Refiner (Tensore unico)
        print(" Loading 3D Models for Refiner (GPU)...")
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        max_id = 16
        num_pts = self.cfg['num_points_mesh']
        all_models = torch.zeros((max_id, num_pts, 3), dtype=torch.float32)
        
        for oid in self.DRS.keys():
            path = os.path.join(models_dir, f"obj_{oid:02d}.ply")
            if os.path.exists(path):
                ply = PlyData.read(path)
                v = ply['vertex']
                pts = np.stack([v['x'], v['y'], v['z']], axis=-1) / 1000.0
                if pts.shape[0] > num_pts:
                    idx = np.random.choice(pts.shape[0], num_pts, replace=False)
                    pts = pts[idx, :]
                all_models[oid] = torch.from_numpy(pts).float()
        return all_models.to(self.device)

    def _load_3d_models_cpu(self):
        # Carica i punti completi per calcolo ADD (Dizionario)
        models = {}
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        for oid in self.DRS.keys():
            path = os.path.join(models_dir, f"obj_{oid:02d}.ply")
            ply = PlyData.read(path)
            v = ply['vertex']
            pts = np.stack([v['x'], v['y'], v['z']], axis=-1) / 1000.0
            models[oid] = pts
        return models

    def quaternion_to_matrix(self, quaternions):
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack(
            (1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
             two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
             two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
        return o.reshape(quaternions.shape[:-1] + (3, 3))
    
    def compute_geodesic_distance(self, R1, R2):
        """Calcola l'errore angolare in gradi tra due matrici di rotazione."""
        R_diff = np.dot(R1, R2.T)
        trace = np.trace(R_diff)
        trace = np.clip(trace, -1, 3) 
        angle = np.arccos((trace - 1) / 2)
        return np.rad2deg(angle)

    def run(self):
        print("\n Starting FINAL METRICS Evaluation (Errors + Accuracy)...")
        
        class_stats = {
            oid: {
                'total': 0,
                'correct': 0,   # Contatore per Accuracy
                'err_deg': [],  # Errore Rotazione (Gradi)
                'err_cm': [],   # Errore Traslazione (cm)
                'add_err': []   # Errore ADD (cm)
            } for oid in self.DRS.keys()
        }
        
        NUM_ITERATIONS = 2

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Eval"):
                img = batch['image'].to(self.device)
                depth = batch['depth'].to(self.device)
                
                gt_t_tensor = batch['translation']
                gt_q_tensor = batch['quaternion']
                gt_t = gt_t_tensor.numpy()[0]
                gt_q = gt_q_tensor.numpy()[0]
                
                obj_id = int(batch['class_id'])
                if obj_id not in self.models_3d_cpu: continue

                # 1. Stima
                pred_r_quat, pred_t, emb_global = self.model_main.forward_refine(img, depth)
                
                # 2. Refinement
                curr_r_quat = pred_r_quat
                curr_t = pred_t
                points_model = self.models_3d_gpu[obj_id].unsqueeze(0)

                for _ in range(NUM_ITERATIONS):
                    R_curr = self.quaternion_to_matrix(curr_r_quat)
                    rotated_pts = torch.bmm(R_curr, points_model.permute(0, 2, 1))
                    cloud_input = rotated_pts + curr_t.unsqueeze(2)
                    delta_r, delta_t = self.refiner(cloud_input, emb_global)
                    curr_t = curr_t + delta_t
                    curr_r_quat = curr_r_quat + delta_r
                    curr_r_quat = torch.nn.functional.normalize(curr_r_quat, p=2, dim=1)

                # 3. Metriche
                pred_R_np = self.quaternion_to_matrix(curr_r_quat).cpu().numpy()[0]
                pred_t_np = curr_t.cpu().numpy()[0]
                gt_R_np = R.from_quat(np.concatenate([gt_q[1:], gt_q[0:1]])).as_matrix()
                pts_full = self.models_3d_cpu[obj_id]

                # A. Errori Fisici
                deg_err = self.compute_geodesic_distance(pred_R_np, gt_R_np)
                cm_err = np.linalg.norm(pred_t_np - gt_t) * 100.0

                # B. ADD (Valore assoluto in cm)
                add_val_m = self.metric_calculator.calculate_metric(pred_R_np, pred_t_np, gt_R_np, gt_t, pts_full, obj_id)
                add_val_cm = add_val_m * 100.0

                # C. Accuracy Check (Soglia 10% diametro)
                diameter_m = self.DRS[obj_id] / 1000.0
                if add_val_m < (diameter_m * 0.1):
                    class_stats[obj_id]['correct'] += 1

                # Store
                class_stats[obj_id]['total'] += 1
                class_stats[obj_id]['err_deg'].append(deg_err)
                class_stats[obj_id]['err_cm'].append(cm_err)
                class_stats[obj_id]['add_err'].append(add_val_cm)

        self._print_final_table(class_stats)
        # Il plot rimane quello degli errori fisici (più utile per debug)
        self._plot_final_results(class_stats)

    def _print_final_table(self, stats):
        print("\n" + "="*100)
        # Tabella ampliata con Accuracy
        print(f"{'OBJECT':<15} | {'ROT Err (°)':<12} | {'TRANS Err (cm)':<15} | {'ADD Err (cm)':<15} | {'ACCURACY (%)':<15}")
        print("-" * 100)
        
        all_deg, all_cm, all_add = [], [], []
        total_correct = 0
        total_samples = 0

        for oid in sorted(stats.keys()):
            s = stats[oid]
            if s['total'] == 0: continue
            
            m_deg = np.mean(s['err_deg'])
            m_cm = np.mean(s['err_cm'])
            m_add = np.mean(s['add_err'])
            acc = (s['correct'] / s['total']) * 100.0
            
            all_deg.extend(s['err_deg'])
            all_cm.extend(s['err_cm'])
            all_add.extend(s['add_err'])
            total_correct += s['correct']
            total_samples += s['total']
            
            name = self.OBJ_NAMES.get(oid, str(oid))
            print(f"{name:<15} | {m_deg:<12.2f} | {m_cm:<15.2f} | {m_add:<15.2f} | {acc:<15.2f}")

        print("="*100)
        global_acc = (total_correct / total_samples * 100.0) if total_samples > 0 else 0
        print(f"{'GLOBAL MEAN':<15} | {np.mean(all_deg):<12.2f} | {np.mean(all_cm):<15.2f} | {np.mean(all_add):<15.2f} | {global_acc:<15.2f}")
        print("="*100)