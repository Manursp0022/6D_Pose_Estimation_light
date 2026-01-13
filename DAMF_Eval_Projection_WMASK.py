import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R
import cv2
from torchvision import transforms
from ultralytics import YOLO

from models.DFMasked_DualAtt_NetVar import DenseFusion_Masked_DualAtt_NetVar
from utils.Posenet_utils.posenet_dataset_AltMasked import LineModPoseDataset_AltMasked
from utils.Posenet_utils.utils_geometric import crop_square_resize


class PointCloudVisualizer:
    """
    Simple visualizer to project 3D point clouds onto RGB images.
    Shows only a few samples (up to 10) for quick inspection.
    NOW WITH RANDOM SAMPLING!
    """
    
    def __init__(self, config):
        self.cfg = config
        self.device = self._get_device()
        print(f"🔧 Initializing Point Cloud Visualizer on: {self.device}")

        # Object diameters (mm)
        self.DIAMETERS = {
            1: 102.09, 2: 247.50, 4: 172.49, 5: 201.40, 6: 154.54,
            8: 261.47, 9: 108.99, 10: 164.62, 11: 175.88, 12: 145.54,
            13: 278.07, 14: 282.60, 15: 212.35
        }
        
        # Object names
        self.OBJ_NAMES = {
            1: "Ape", 2: "Benchvise", 4: "Cam", 5: "Can", 6: "Cat", 
            8: "Driller", 9: "Duck", 10: "Eggbox", 11: "Glue", 
            12: "Holepuncher", 13: "Iron", 14: "Lamp", 15: "Phone"
        }

        self.LINEMOD_TO_YOLO = {
            1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 9: 6,
            10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12
        }

        # Camera intrinsics (LineMOD)
        self.cam_params_norm = torch.tensor([
            572.4114 / 640, 573.57043 / 480,
            325.2611 / 640, 242.04899 / 480
        ], dtype=torch.float32)
        
        # Full camera intrinsics for projection
        self.fx = 572.4114
        self.fy = 573.57043
        self.cx = 325.2611
        self.cy = 242.04899
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        self.img_h, self.img_w = 480, 640
        self.img_size = 224

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Setup
        self.yolo_model = self._setup_yolo()
        self.dataset = self._setup_dataset()
        self.models_3d = self._load_3d_models()
        self.model = self._setup_model()
        
        self.YOLO_CONF = 0.5
        
        # Output directory
        self.output_dir = os.path.join(self.cfg.get('save_dir', 'results'), 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _setup_yolo(self):
        print("🔍 Loading YOLO model...")
        yolo_path = self.cfg['yolo_model_path']
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"❌ YOLO model not found at: {yolo_path}")
        model = YOLO(yolo_path)
        print(f"   ✓ Loaded YOLO from: {yolo_path}")
        return model

    def _setup_dataset(self):
        print("📦 Loading Dataset...")
        dataset = LineModPoseDataset_AltMasked(
            self.cfg['dataset_root'], 
            mode='val'
        )
        print(f"   ✓ Loaded {len(dataset)} samples")
        return dataset

    def _setup_model(self):
        print("🧠 Loading model...")
        model = DenseFusion_Masked_DualAtt_NetVar(
            pretrained=False,
            temperature=self.cfg.get('temperature', 1.5)
        ).to(self.device)
        
        weights_path = os.path.join(
            self.cfg['model_dir'], 
            'DenseFusion_Masked_DualAtt_NetVar_WOAttention_Hard.pth'
        )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"❌ Weights not found at: {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove compile prefix if present
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"   ✓ Loaded weights from: {weights_path}")
        return model
    
    def _load_3d_models(self):
        print("📐 Loading 3D models...")
        models_3d = {}
        models_dir = os.path.join(self.cfg['dataset_root'], 'models')
        
        for obj_id in self.DIAMETERS.keys():
            path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
            if not os.path.exists(path):
                continue
                
            ply = PlyData.read(path)
            vertex = ply['vertex']
            pts_mm = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
            models_3d[obj_id] = pts_mm / 1000.0  # mm -> m
        
        print(f"   ✓ Loaded {len(models_3d)} 3D models")
        return models_3d

    def _quaternion_to_matrix(self, quat):
        """Convert single quaternion [w, x, y, z] to rotation matrix."""
        if isinstance(quat, torch.Tensor):
            quat = quat.cpu().numpy()
        
        # scipy wants [x, y, z, w]
        quat_scipy = np.concatenate([quat[1:], quat[0:1]])
        return R.from_quat(quat_scipy).as_matrix()

    def _project_points(self, points_3d, R_mat, t):
        """
        Project 3D points to 2D image.
        
        Args:
            points_3d: [N, 3] points in meters
            R_mat: [3, 3] rotation matrix
            t: [3] translation in meters
            
        Returns:
            points_2d: [N, 2] pixel coordinates
            valid_mask: [N] bool array
        """
        # Transform to camera coordinates
        points_cam = (R_mat @ points_3d.T).T + t
        
        # Filter points behind camera
        valid_mask = points_cam[:, 2] > 0
        
        # Project to image plane
        points_2d = (self.K @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        # Filter out of bounds
        in_bounds = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < self.img_w) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < self.img_h)
        )
        valid_mask = valid_mask & in_bounds
        
        return points_2d, valid_mask

    def _process_sample(self, idx):
        """
        Process a single sample and return all needed data.
        
        Returns:
            dict with rgb_img, pred_R, pred_t, gt_R, gt_t, points_3d, obj_id
            or None if processing failed
        """
        # Get sample from dataset
        sample = self.dataset[idx]
        
        rgb_path = sample['path']
        depth_path = sample['depth_path']
        obj_id = int(sample['class_id'])
        gt_quat = sample['quaternion'].numpy()
        gt_trans = sample['translation'].numpy()
        
        # Load original RGB
        rgb_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        
        # Run YOLO
        target_yolo_class = self.LINEMOD_TO_YOLO[obj_id]
        yolo_results = self.yolo_model(rgb_path, conf=self.YOLO_CONF, verbose=False)
        yolo_res = yolo_results[0]
        
        # Find best detection
        boxes = yolo_res.boxes
        best_idx = None
        best_conf = 0.0
        for j, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
            if int(cls) == target_yolo_class and float(conf) > best_conf:
                best_conf = float(conf)
                best_idx = j
        
        if best_idx is None:
            print(f"   ⚠️  YOLO missed object in sample {idx}")
            return None
        
        # Load depth
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        
        # Get bbox
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        bbox = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
        
        # Get mask
        mask_data = yolo_res.masks.data[best_idx].cpu().numpy()
        if mask_data.shape != (self.img_h, self.img_w):
            mask_data = cv2.resize(mask_data, (self.img_w, self.img_h), 
                                 interpolation=cv2.INTER_NEAREST)
        mask = (mask_data > 0.5).astype(np.uint8) * 255
        
        # Crop & resize
        rgb_crop = crop_square_resize(rgb_img, bbox, self.img_size, is_depth=False)
        depth_crop = crop_square_resize(depth_img, bbox, self.img_size, is_depth=True)
        mask_crop = crop_square_resize(mask, bbox, self.img_size, is_depth=True)
        mask_crop = (mask_crop > 127).astype(np.float32)
        
        # To tensors
        rgb_tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
        depth_tensor = torch.from_numpy(depth_crop).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask_crop).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Bbox normalized
        x, y, w, h = bbox
        bbox_norm = torch.tensor([[
            (x + w/2) / self.img_w,
            (y + h/2) / self.img_h,
            w / self.img_w,
            h / self.img_h
        ]], dtype=torch.float32).to(self.device)
        
        # Camera params
        cam_params = self.cam_params_norm.unsqueeze(0).to(self.device)
        
        # Model inference
        with torch.no_grad():
            pred_quat, pred_trans = self.model(rgb_tensor, depth_tensor, bbox_norm, 
                                              cam_params, mask_tensor)
        
        # Convert to numpy
        pred_quat = pred_quat.squeeze(0).cpu().numpy()
        pred_trans = pred_trans.squeeze(0).cpu().numpy()
        
        # Convert quaternions to rotation matrices
        pred_R = self._quaternion_to_matrix(pred_quat)
        gt_R = self._quaternion_to_matrix(gt_quat)
        
        # Get 3D model
        if obj_id not in self.models_3d:
            print(f"   ⚠️  No 3D model for obj_id={obj_id}")
            return None
        
        points_3d = self.models_3d[obj_id]
        
        return {
            'rgb_img': rgb_img,
            'pred_R': pred_R,
            'pred_t': pred_trans,
            'gt_R': gt_R,
            'gt_t': gt_trans,
            'points_3d': points_3d,
            'obj_id': obj_id
        }

    def visualize_sample(self, data, sample_idx):
        """
        Create and save visualization for one sample showing ONLY predictions.
        
        Args:
            data: dict from _process_sample
            sample_idx: sample index for filename
        """
        rgb_img = data['rgb_img']
        pred_R = data['pred_R']
        pred_t = data['pred_t']
        points_3d = data['points_3d']
        obj_id = data['obj_id']
        
        # Project only predicted pose
        pred_pts_2d, pred_valid = self._project_points(points_3d, pred_R, pred_t)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Display image
        ax.imshow(rgb_img)
        
        # Overlay Predicted point cloud with increased transparency
        if pred_valid.any():
            # alpha=0.3 makes points more transparent
            # s=1 makes points smaller to avoid cluttering the object details
            ax.scatter(pred_pts_2d[pred_valid, 0], pred_pts_2d[pred_valid, 1], 
                      c='cyan', s=1, alpha=0.3, label='Predicted Pose', edgecolors='none')
        
        # Title and labels
        obj_name = self.OBJ_NAMES.get(obj_id, f"Object {obj_id}")
        ax.set_title(f'Prediction: {obj_name} (Sample {sample_idx})', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.legend(loc='upper right', fontsize=12)
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.output_dir, 
                                f'pred_{obj_name.lower()}_{sample_idx:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"    ✓ Saved Prediction: {save_path}")
        return save_path

    def run(self, num_samples=10, random_samples=True):
        """
        Visualize point cloud projections for a few samples.
        
        Args:
            num_samples: Number of samples to visualize (default: 10)
            random_samples: If True, select random samples; if False, start from beginning (default: True)
        """
        print("\n" + "="*70)
        print("🎨 POINT CLOUD PROJECTION VISUALIZER")
        print("="*70)
        print(f"📊 Will visualize {num_samples} samples")
        print(f"🎲 Random selection: {random_samples}")
        print(f"💾 Output directory: {self.output_dir}")
        print("="*70 + "\n")
        
        self.model.eval()
        
        # Select indices to process
        if random_samples:
            # Randomly select indices
            np.random.seed(None)  # Use current time for randomness
            total_samples = len(self.dataset)
            selected_indices = np.random.choice(total_samples, 
                                               size=min(num_samples * 3, total_samples), 
                                               replace=False)
            print(f"🎲 Randomly selected {len(selected_indices)} candidate samples from {total_samples} total")
        else:
            # Sequential from start
            selected_indices = list(range(min(num_samples * 3, len(self.dataset))))
        
        saved_count = 0
        
        for current_idx in selected_indices:
            if saved_count >= num_samples:
                break
                
            print(f"Processing sample {current_idx}...")
            
            # Process sample
            data = self._process_sample(current_idx)
            
            if data is not None:
                # Visualize and save
                self.visualize_sample(data, current_idx)
                saved_count += 1
        
        print("\n" + "="*70)
        print(f"✅ COMPLETE: Saved {saved_count} visualizations")
        print(f"📁 Location: {self.output_dir}")
        print("="*70 + "\n")
        
        return self.output_dir


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    config = {
        'dataset_root': './dataset/Linemod_preprocessed',
        'model_dir': './checkpoints',
        'yolo_model_path': 'C:/Users/gabri/Desktop/AML project/6D_Pose_Estimation_light/checkpoints/final_best_seg_YOLO.pt',
        'save_dir': './results',
        'temperature': 1.5,
    }
    
    # Create visualizer
    visualizer = PointCloudVisualizer(config)
    
    # Visualize 10 RANDOM samples (NEW DEFAULT!)
    visualizer.run(num_samples=10, random_samples=True)
    
    # Or sequential samples (old behavior)
    # visualizer.run(num_samples=10, random_samples=False)