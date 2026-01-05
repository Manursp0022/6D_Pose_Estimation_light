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
from models.Posenet import PoseResNet
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
from utils.Posenet_utils.utils_geometric import solve_pinhole_diameter
from utils.Posenet_utils.PoseEvaluator import PoseEvaluator 
from utils.Posenet_utils.utils_geometric import crop_square_resize

class PoseNetEvaluator:
    def __init__(self, config):
        """
        Handles the complete model evaluation (ADD Metric) and plotting.
        """
        self.cfg = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using NVIDIA CUDA")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple MPS (Metal Acceleration)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        print(f"Initializing Evaluator on: {self.device}")

        # Diameter Dictionary (in mm) for Pinhole calculations and Thresholds
        self.DRS = {
            1: 102.09, 2: 247.50, 4: 172.49, 5: 201.40, 6: 154.54, 8: 261.47,
            9: 108.99, 10: 164.62, 11: 175.88, 12: 145.54, 13: 278.07, 14: 282.60, 15: 212.35
        }
        
        # ID to Name Mapping (for better plot readability)
        self.OBJ_NAMES = {
            1: "Ape", 2: "Benchvise", 4: "Cam", 5: "Can", 6: "Cat", 8: "Driller",
            9: "Duck", 10: "Eggbox", 11: "Glue", 12: "Holepuncher", 13: "Iron", 14: "Lamp", 15: "Phone"
        }

        self.val_loader = self._setup_data()
        self.Resnet = self._setup_Resnet()
        self.refiner = self._setup_Refiner()
        self.YOLO = self._setup_YOLO()
        self.models_3d = self._load_3d_models()
        
        # Instance of YOUR metric calculation class
        # We pass a dummy K matrix because we only need the 'calculate_metric' method
        self.metric_calculator = PoseEvaluator(np.eye(3))

    def _setup_data(self):
        print(" Loading Validation Dataset...")
        val_ds = LineModPoseDataset(self.cfg['split_val'], self.cfg['dataset_root'], mode='val')
        loader = DataLoader(val_ds, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=12)
        print(f"   -> Found {len(val_ds)} samples.")
        return loader

    def _setup_Resnet(self):
        print(" Loading ResNet Model...")
        model = PoseResNet(pretrained=False).to(self.device)
        
        weights_path = os.path.join(self.cfg['model_dir'], 'best_posenet_baseline.pth')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f" Resnet Weights not found at: {weights_path}")
        
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        print("   -> Weights loaded successfully.")
        return model
    
    def _setup_Refiner(self):
        print(" Loading Pinhole Refiner Model...")
        from Refinement_Section.Pinhole_Refinement import PinholeRefineNet
        refiner = PinholeRefineNet().to(self.device)
        
        weights_path = os.path.join(self.cfg['model_dir'], 'best_pinhole_refiner.pth')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f" Refiner Weights not found at: {weights_path}")
        
        refiner.load_state_dict(torch.load(weights_path, map_location=self.device))
        refiner.eval()
        print("   -> Weights loaded successfully.")
        return refiner

    def _setup_YOLO(self):
        print("Loading YOLO model....")
        weights_path = os.path.join(self.cfg['model_dir'], 'best_YOLO.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f" YOLO Weights not found at: {weights_path}")

        model = YOLO(weights_path).to(self.device)
        return model

    def _load_3d_models(self):
        """Loads .ply meshes into memory for ADD calculation."""
        print(" Loading 3D Models (.ply)...")
        models_3d = {}
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        
        if not os.path.exists(models_dir):
            print(f" WARNING: Models directory not found: {models_dir}")
            return {}

        count = 0
        for obj_id in self.DRS.keys():
            filename = f"obj_{obj_id:02d}.ply"
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                ply = PlyData.read(path)
                vertex = ply['vertex']
                #from millimeters to meters 
                pts_mm = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
                models_3d[obj_id] = pts_mm / 1000.0 
                
                count += 1
        
        print(f"   -> Loaded {count} 3D models.")
        return models_3d

    def _quaternion_to_matrix(self, quats):
        """Helper: Batch Quaternions -> Batch Rotation Matrices"""
        if isinstance(quats, torch.Tensor):
            quats = quats.cpu().numpy()
        
        # Reorder [w, x, y, z] (PyTorch output) -> [x, y, z, w] (SciPy expectation)
        quats_scipy = np.concatenate([quats[:, 1:], quats[:, 0:1]], axis=1)
        return R.from_quat(quats_scipy).as_matrix()

    def run(self):
        print("\n Starting ADD Evaluation...")

        # LINEMOD ID: YOLO ID
        lm_to_yolo = {
            1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 
            9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12
        }

        resnet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) #ResNet eat only normalized tensors
        
        # Dictionary to store per-class statistics
        class_stats = {oid: {'correct': 0, 'total': 0, 'errors': []} for oid in self.DRS.keys()}

        all_trans_errors = []
        all_rot_errors_m = []
        all_rot_errors_deg = []
        
        total_correct = 0
        total_preds = 0
        all_errors = [] # Questo rimane l'ADD combinato
        YOLO_CONF = 0.5
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Dati Ground Truth
                paths = batch['path']
                # = batch['bbox'].to(self.device)
                intrinsics = batch['cam_params'].to(self.device)
                gt_translation = batch['translation'].to(self.device)
                gt_quats = batch['quaternion'].to(self.device)
                class_ids = batch['class_id'].numpy()

                # YOLO PREDICTION 
                # Passiamo i path: è più lento ma sicuro (evita problemi di normalizzazione colori)
                yolo_results = self.YOLO(paths, conf=YOLO_CONF,verbose=False)

                # Prepariamo i batch per Pinhole e ResNet
                pred_bboxes_list = []
                resnet_input_list = []
                valid_indices = [] # Indici del batch dove YOLO ha trovato l'oggetto

                for i, result in enumerate(yolo_results):
                    target_linemod_id = int(class_ids[i])                    

                    target_yolo_id = lm_to_yolo[target_linemod_id] # Es. Driller(8) -> 5
                    
                    best_conf = -1
                    found_box = None

                    for box in result.boxes:
                        cls = int(box.cls[0]) 
                        conf = float(box.conf[0])
                        
                        # VERIFICA MAPPING ID: Se YOLO è trainato su classi 0-12 e LINEMOD è 1-15, controlla!
                        # Qui assumo corrispondenza diretta.
                        if cls == target_yolo_id and conf > best_conf:
                            # YOLO format xyxy -> convertiamo in x,y,w,h per la tua funzione crop
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            w = x2 - x1
                            h = y2 - y1
                            found_box = torch.tensor([x1, y1, w, h], dtype=torch.float32)
                            best_conf = conf
                    
                    if found_box is not None:
                        valid_indices.append(i)
                        pred_bboxes_list.append(found_box)

                        img_raw = cv2.imread(paths[i])
                        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                        
                        # cropping using YOLO prediction
                        img_crop = crop_square_resize(img_raw, found_box, 224) # 224 hardcoded o self.cfg
                        
                        # Trasformazione (ToTensor + Normalize)
                        img_tensor = resnet_transform(img_crop)
                        resnet_input_list.append(img_tensor)
                    else:
                        # YOLO non ha trovato l'oggetto -> Errore massimo (Detection Failure)
                        # Non possiamo calcolare ADD, lo contiamo come Fail.
                        pass

                if len(valid_indices) == 0:
                    continue

                #creating tensors only with found objects
                pred_bboxes_tensor = torch.stack(pred_bboxes_list).to(self.device)
                resnet_batch = torch.stack(resnet_input_list).to(self.device)
                #refiner_batch = 
                # Filtriamo le Ground Truth corrispondenti agli oggetti trovati
                # (Dobbiamo confrontare solo quelli che YOLO ha visto)
                subset_intrinsics = intrinsics[valid_indices]
                subset_gt_trans = gt_translation[valid_indices]
                subset_gt_quats = gt_quats[valid_indices]
                subset_class_ids = class_ids[valid_indices]
                
                # Pinhole 
                current_diameters = [self.DRS[cid] / 1000.0 for cid in subset_class_ids]
                diam_tensor = torch.tensor(current_diameters, dtype=torch.float32).to(self.device)
                pred_trans = solve_pinhole_diameter(pred_bboxes_tensor, subset_intrinsics, diam_tensor)
               
                pred_trans = self.refiner(pred_trans, pred_bboxes_tensor)
               
                pred_quats = self.Resnet(resnet_batch)

                # Calculate Metrics (CPU side)
                pred_trans_np = pred_trans.cpu().numpy()
                #Using subset (Important: covering the case in which YOLO finds nothing in some images)
                gt_trans_np = subset_gt_trans.cpu().numpy() 
                
                pred_R_np = self._quaternion_to_matrix(pred_quats)
                gt_R_np = self._quaternion_to_matrix(subset_gt_quats)

                for i, obj_id in enumerate(subset_class_ids):
                    obj_id = int(obj_id)
                    
                    if obj_id not in self.models_3d:
                        continue # Skip if 3D model is missing

                    pts_3d = self.models_3d[obj_id]
                    
                    # Calculate ADD/ADD-S using your PoseEvaluator class
                    add_error_m = self.metric_calculator.calculate_metric(
                        pred_R_np[i], pred_trans_np[i], 
                        gt_R_np[i], gt_trans_np[i], 
                        pts_3d, obj_id
                    )

                    t_err, r_err_m, r_err_deg, tx, ty, tz = self.metric_calculator.calculate_separated_metrics(
                        pred_R_np[i], pred_trans_np[i], 
                        gt_R_np[i], gt_trans_np[i], 
                        pts_3d, obj_id
                    )

                    # Accumulo statistiche
                    all_trans_errors.append(t_err * 100.0)      # Convert to cm
                    all_rot_errors_m.append(r_err_m * 100.0)    # Convert to cm
                    all_rot_errors_deg.append(r_err_deg)
                    
                    # Check Correctness (Threshold: 10% of diameter)
                    diameter_m = self.DRS[obj_id] / 1000.0
                    threshold = 0.5 * diameter_m
                    
                    is_correct = self.metric_calculator.is_pose_correct(add_error_m, threshold)
                    
                    # Global Stats Update
                    if is_correct:
                        total_correct += 1
                    total_preds += 1
                    all_errors.append(add_error_m * 100.0) # Store in cm

                    # Per-Class Stats Update
                    if obj_id in class_stats:
                        class_stats[obj_id]['total'] += 1
                        class_stats[obj_id]['errors'].append(add_error_m * 100.0)
                        if is_correct:
                            class_stats[obj_id]['correct'] += 1

        # Final Text Report
        accuracy = (total_correct / total_preds * 100.0) if total_preds > 0 else 0.0
        mean_add = np.mean(all_errors) if len(all_errors) > 0 else 0.0

        # Medie separate
        mean_trans = np.mean(all_trans_errors) if len(all_trans_errors) > 0 else 0.0
        mean_rot_cm = np.mean(all_rot_errors_m) if len(all_rot_errors_m) > 0 else 0.0
        mean_rot_deg = np.mean(all_rot_errors_deg) if len(all_rot_errors_deg) > 0 else 0.0
        
        self._print_report(accuracy, mean_add, mean_trans, mean_rot_cm, mean_rot_deg, total_preds)
        
        # Generate Plots
        self._plot_per_class_results(class_stats)

    def _print_report(self, accuracy, mean_add, mean_trans, mean_rot_cm, mean_rot_deg, total):
        print("\n" + "="*60)
        print("FINAL REPORT (ADD Metric & Separated Breakdown)")
        print("="*60)
        print(f" Samples Evaluated: {total}")
        print("-" * 40)
        print(f" TOTAL ACCURACY:    {accuracy:.2f} %")
        print(f" COMBINED ADD Error:{mean_add:.2f} cm")
        print("-" * 40)
        print(" ERROR BREAKDOWN:")
        print(f" -> Translation Err: {mean_trans:.2f} cm")
        print(f" -> Rotation Err:    {mean_rot_cm:.2f} cm (Displacement on mesh)")
        print(f" -> Rotation Angle:  {mean_rot_deg:.2f} deg")
        print("="*60)

    def _plot_per_class_results(self, class_stats):
        """Generates two plots: Accuracy per Class and Mean Error vs Threshold"""
        print(" Generating Per-Class Plots...")
        
        # Prepare data for plotting
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
            
            # Threshold in cm (D/10)
            # DRS is in mm, so: (mm / 1000) * 0.1 * 100 -> cm
            thresholds_cm.append((self.DRS[oid] / 1000.0 * 0.5) * 100.0)

        # Create Figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # --- PLOT 1: Accuracy (%) ---
        bars = ax1.bar(labels, accuracies, color='skyblue', edgecolor='black')
        ax1.set_ylabel('Accuracy (% of Pass)')
        ax1.set_title('Per-Class Accuracy (ADD < 0.5 Diameter)')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        # --- PLOT 2: Mean Error vs Threshold ---
        x = np.arange(len(labels))
        width = 0.35
        
        # Mean Error Bar
        rects1 = ax2.bar(x - width/2, mean_errors, width, label='Mean ADD Error (cm)', color='salmon')
        # Threshold Bar (for direct comparison)
        rects2 = ax2.bar(x + width/2, thresholds_cm, width, label='Threshold (0.5*D)', color='lightgreen', alpha=0.7)
        
        ax2.set_ylabel('Distance (cm)')
        ax2.set_title('Mean ADD Error vs Threshold per Object')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        
        # Save Plot
        save_path = os.path.join(self.cfg['save_dir'], 'per_class_results.png')
        plt.savefig(save_path)
        print(f"📈 Plot saved to: {save_path}")
        plt.show()
if __name__ == "__main__":
    # Example configuration dictionary
    config = {
        'split_val': 'data/autosplit_val_ALL.txt',
        'dataset_root': 'dataset/Linemod_preprocessed',
        'model_dir': 'checkpoints/',
        'batch_size': 16,
        'save_dir': 'checkpoints_results/'
    }

    evaluator = PoseNetEvaluator(config)
    evaluator.run()