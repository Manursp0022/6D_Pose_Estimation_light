import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

from Refinement_Section.ICP_Refiner import ICPRefiner
from utils.Posenet_utils.utils_geometric import matrix_to_quaternion, quaternion_to_matrix_np

# Check Open3D availability
try:
    import open3d as o3d
    HAS_OPEN3D = True
    print("✅ Open3D available for ICP")
except:
    HAS_OPEN3D = False
    print("⚠️  Open3")

class ICPEvaluator:
    """
    Evaluator that compares Base Model vs Base Model + ICP Refinement.
    """
    
    def __init__(self, config):
        self.cfg = config
        self.device = self._get_device()
        print(f"🔧 Initializing ICP Evaluator on: {self.device}")

        # LineMOD object diameters (mm)
        self.DIAMETERS = {
            1: 102.09, 2: 247.50, 4: 172.49, 5: 201.40, 6: 154.54,
            8: 261.47, 9: 108.99, 10: 164.62, 11: 175.88, 12: 145.54,
            13: 278.07, 14: 282.60, 15: 212.35
        }
        
        self.OBJ_NAMES = {
            1: "Ape", 2: "Benchvise", 4: "Cam", 5: "Can", 6: "Cat", 
            8: "Driller", 9: "Duck", 10: "Eggbox", 11: "Glue", 
            12: "Holepuncher", 13: "Iron", 14: "Lamp", 15: "Phone"
        }
        
        # Symmetric objects (use ADD-S metric)
        self.SYMMETRIC_OBJECTS = {10, 11}  # Eggbox, Glue

        # Setup components
        self.val_loader = self._setup_data()
        self.models_3d = self._load_3d_models()
        self.base_model = self._setup_model()
        self.icp_refiner = self._setup_icp()

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _setup_data(self):
        """Load validation dataset."""
        from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
        
        print("📦 Loading Validation Dataset...")
        val_ds = LineModPoseDataset(
            self.cfg['split_val'], 
            self.cfg['dataset_root'], 
            mode='val'
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.cfg.get('batch_size', 1),  # Batch=1 for ICP
            shuffle=False, 
            num_workers=self.cfg.get('num_workers', 0),
            pin_memory=False
        )
        
        print(f"   ✓ Loaded {len(val_ds)} validation samples")
        return val_loader

    def _load_3d_models(self):
        """Load 3D models for ADD metric and ICP."""
        print("📐 Loading 3D Models (.ply)...")
        models_3d = {}
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        
        for obj_id in self.DIAMETERS.keys():
            path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
            
            if not os.path.exists(path):
                print(f"   ⚠️  Missing: obj_{obj_id:02d}.ply")
                continue
                
            ply = PlyData.read(path)
            vertex = ply['vertex']
            pts_mm = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
            models_3d[obj_id] = pts_mm / 1000.0  # mm -> m
        
        print(f"   ✓ Loaded {len(models_3d)} 3D models")
        return models_3d

    def _setup_model(self):
        """Load the base pose estimation model."""
        from models.DFMasked_DualAtt_NetVar import DenseFusion_Masked_DualAtt_NetVar
        
        print("🧠 Loading Base Model...")
        
        model = DenseFusion_Masked_DualAtt_NetVar(
            pretrained=False,
            temperature=self.cfg.get('temperature', 2.0)
        ).to(self.device)
        
        weights_path = os.path.join(self.cfg['model_dir'], self.cfg['base_weights'])
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove torch.compile prefix
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"   ✓ Loaded from: {weights_path}")
        return model

    def _setup_icp(self):
        """Setup ICP refiner."""
        print("🔧 Setting up ICP Refiner...")
        
        # Camera intrinsics for LineMOD
        cam_K = np.array([
            [572.4114, 0, 325.2611],
            [0, 573.57043, 242.04899],
            [0, 0, 1]
        ])
        
        # Override with config if provided
        if 'cam_K' in self.cfg:
            cam_K = np.array(self.cfg['cam_K'])
        
        refiner = ICPRefiner(
            models_3d=self.models_3d,
            cam_K=cam_K,
            max_correspondence_distance=self.cfg.get('icp_max_dist', 0.02),
            max_iterations=self.cfg.get('icp_iterations', 50),
            n_model_points=self.cfg.get('icp_n_points', 1000),
            method='open3d' if HAS_OPEN3D else 'cv2'
        )
        
        print(f"   ✓ ICP Refiner ready (method: {'open3d' if HAS_OPEN3D else 'cv2'})")
        return refiner

    def _quaternion_to_matrix(self, quat):
        """Convert quaternion [w,x,y,z] to rotation matrix."""
        if isinstance(quat, torch.Tensor):
            quat = quat.cpu().numpy()
        return quaternion_to_matrix_np(quat)

    def _compute_add(self, pred_R, pred_t, gt_R, gt_t, pts_3d, symmetric=False):
        """Compute ADD or ADD-S error."""
        pred_pts = pts_3d @ pred_R.T + pred_t
        gt_pts = pts_3d @ gt_R.T + gt_t
        
        if symmetric:
            from scipy.spatial import cKDTree
            tree = cKDTree(gt_pts)
            dists, _ = tree.query(pred_pts, k=1)
            return np.mean(dists)
        else:
            return np.mean(np.linalg.norm(pred_pts - gt_pts, axis=1))

    def _get_depth_from_batch(self, depth_tensor):
        """
        Extract depth image from batch tensor.
        Handles different depth formats.
        """
        depth = depth_tensor.squeeze().cpu().numpy()
        
        # If depth is normalized, denormalize (assume max 10m)
        if depth.max() <= 1.0:
            depth = depth * 10.0
        
        # If depth is in mm, convert to m
        if depth.max() > 100:
            depth = depth / 1000.0
        
        return depth

    def _get_mask_from_batch(self, mask_tensor):
        """Extract mask from batch tensor."""
        mask = mask_tensor.squeeze().cpu().numpy()
        return mask > 0.5

    @torch.no_grad()
    def evaluate(self, use_icp=False):
        """
        Run evaluation.
        
        Args:
            use_icp: bool, whether to apply ICP refinement
            
        Returns:
            results dict
        """
        mode = "Base + ICP" if use_icp else "Base Only"
        print(f"\n🚀 Evaluating: {mode}")
        
        total_correct = 0
        total_predictions = 0
        all_errors = []
        icp_stats = {'success': 0, 'fail': 0, 'skipped': 0}
        
        class_stats = {obj_id: {'correct': 0, 'total': 0, 'errors': []} 
                       for obj_id in self.DIAMETERS.keys()}

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # Get data
            rgb = batch['image'].to(self.device)
            depth_tensor = batch['depth'].to(self.device)
            mask_tensor = batch['mask'].to(self.device)
            bb_info = batch['bbox_norm'].to(self.device)
            cam_params = batch['cam_params'].to(self.device)
            gt_quat = batch['quaternion']
            gt_trans = batch['translation']
            class_ids = batch['class_id']
            
            # Get base model prediction
            pred_quat, pred_trans = self.base_model(
                rgb, depth_tensor, bb_info, cam_params, mask_tensor
            )
            
            # Process each sample in batch
            batch_size = rgb.size(0)
            
            for i in range(batch_size):
                obj_id = int(class_ids[i])
                
                if obj_id not in self.models_3d:
                    continue
                
                # Convert predictions to numpy
                pred_q = pred_quat[i].cpu().numpy()
                pred_t = pred_trans[i].cpu().numpy()
                gt_q = gt_quat[i].numpy()
                gt_t = gt_trans[i].numpy()
                
                # Convert quaternions to rotation matrices
                pred_R = self._quaternion_to_matrix(pred_q)
                gt_R = self._quaternion_to_matrix(gt_q)
                
                # Apply ICP refinement if enabled
                if use_icp:
                    depth = self._get_depth_from_batch(depth_tensor[i])
                    mask = self._get_mask_from_batch(mask_tensor[i])
                    
                    # Run ICP
                    refined_R, refined_t, success = self.icp_refiner.refine(
                        pred_R, pred_t, depth, mask, obj_id,
                        min_points=self.cfg.get('icp_min_points', 300)
                    )
                    
                    if success:
                        pred_R = refined_R
                        pred_t = refined_t
                        icp_stats['success'] += 1
                    else:
                        icp_stats['fail'] += 1
                
                # Compute ADD error
                pts_3d = self.models_3d[obj_id]
                is_symmetric = obj_id in self.SYMMETRIC_OBJECTS
                
                add_error = self._compute_add(
                    pred_R, pred_t, gt_R, gt_t, pts_3d, symmetric=is_symmetric
                )
                
                # Check threshold
                diameter_m = self.DIAMETERS[obj_id] / 1000.0
                threshold_m = 0.1 * diameter_m
                is_correct = add_error < threshold_m
                
                # Update stats
                if is_correct:
                    total_correct += 1
                total_predictions += 1
                all_errors.append(add_error * 100)  # cm
                
                class_stats[obj_id]['total'] += 1
                class_stats[obj_id]['errors'].append(add_error)
                if is_correct:
                    class_stats[obj_id]['correct'] += 1
        
        # Compute final metrics
        accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
        mean_error = np.mean(all_errors) if all_errors else 0
        
        results = {
            'accuracy': accuracy,
            'mean_error_cm': mean_error,
            'total_correct': total_correct,
            'total_predictions': total_predictions,
            'class_stats': class_stats,
            'mode': mode,
            'icp_stats': icp_stats if use_icp else None
        }
        
        return results

    def compare(self):
        """Compare Base Only vs Base + ICP."""
        print("\n" + "="*70)
        print("COMPARISON: Base Model vs Base Model + ICP Refinement")
        print("="*70)
        
        # Evaluate without ICP
        results_base = self.evaluate(use_icp=False)
        
        # Evaluate with ICP
        results_icp = self.evaluate(use_icp=True)
        
        # Print comparison
        self._print_comparison(results_base, results_icp)
        
        return {'base': results_base, 'icp': results_icp}

    def _print_comparison(self, results_base, results_icp):
        """Print comparison table."""
        print("\n" + "="*70)
        print("RESULTS COMPARISON")
        print("="*70)
        
        acc_base = results_base['accuracy']
        err_base = results_base['mean_error_cm']
        acc_icp = results_icp['accuracy']
        err_icp = results_icp['mean_error_cm']
        
        acc_imp = acc_icp - acc_base
        err_imp = (err_base - err_icp) / err_base * 100 if err_base > 0 else 0
        
        print(f"\n{'Metric':<25} {'Base Only':<15} {'Base + ICP':<15} {'Change':<15}")
        print("-"*70)
        print(f"{'Accuracy':<25} {acc_base:>12.2f}% {acc_icp:>12.2f}% {acc_imp:>+12.2f}%")
        print(f"{'Mean ADD Error':<25} {err_base:>11.3f}cm {err_icp:>11.3f}cm {err_imp:>+11.1f}%")
        
        # ICP statistics
        if results_icp['icp_stats']:
            stats = results_icp['icp_stats']
            total = stats['success'] + stats['fail']
            success_rate = stats['success'] / total * 100 if total > 0 else 0
            print(f"\n{'ICP Success Rate':<25} {success_rate:>12.1f}%")
            print(f"{'ICP Successes':<25} {stats['success']:>12}")
            print(f"{'ICP Failures':<25} {stats['fail']:>12}")
        
        # Per-class comparison
        print("\n" + "-"*70)
        print(f"{'Object':<15} {'Base':<12} {'+ ICP':<12} {'Change':<12} {'Status'}")
        print("-"*70)
        
        for obj_id in sorted(self.DIAMETERS.keys()):
            name = self.OBJ_NAMES[obj_id]
            
            stats_base = results_base['class_stats'][obj_id]
            stats_icp = results_icp['class_stats'][obj_id]
            
            if stats_base['total'] == 0:
                continue
            
            acc_b = stats_base['correct'] / stats_base['total'] * 100
            acc_i = stats_icp['correct'] / stats_icp['total'] * 100
            change = acc_i - acc_b
            
            if change > 0.5:
                emoji = "✅ improved"
            elif change < -0.5:
                emoji = "❌ worse"
            else:
                emoji = "➖ same"
            
            print(f"{name:<15} {acc_b:>10.2f}% {acc_i:>10.2f}% {change:>+10.2f}% {emoji}")
        
        print("="*70)
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Base Accuracy:     {acc_base:.2f}%")
        print(f"   Base + ICP:        {acc_icp:.2f}%")
        print(f"   Improvement:       {acc_imp:+.2f}%")
        
        if acc_imp > 0:
            print(f"\n   🎉 ICP refinement improved accuracy!")
        elif acc_imp < 0:
            print(f"\n   ⚠️  ICP refinement decreased accuracy (may need tuning)")
        else:
            print(f"\n   ➖ No significant change from ICP")
        
        print("="*70 + "\n")

if __name__ == "__main__":
    print("="*60)
    print("ICP Refinement Evaluation")
    print("="*60)
    print()

    config = {
        # Paths
        'dataset_root': '/content/Linemod_preprocessed',
        'split_val': '/content/6D_Pose_Estimation_light/val_ALL.txt',
        'model_dir': '/content/6D_Pose_Estimation_light/checkpoints',
        'base_weights': 'DenseFusion_Masked_DualAtt_NetVar.pth',
        
        # Data
        'batch_size': 1,  # Must be 1 for ICP (processes each sample individually)
        'num_workers': 0,
        
        # Model
        'temperature': 2.0,
        
        # ICP parameters
        'icp_max_dist': 0.02,      # 2cm max correspondence distance
        'icp_iterations': 50,       # ICP iterations
        'icp_n_points': 1000,       # Model points to use
        'icp_min_points': 300,      # Min points required
        
        # Camera (LineMOD default)
        'cam_K': [
            [572.4114, 0, 325.2611],
            [0, 573.57043, 242.04899],
            [0, 0, 1]
        ]
    }

    evaluator = ICPEvaluator(config)
    # Compare base vs ICP
    results = evaluator.compare()


