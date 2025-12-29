"""
ICP Debug Script

Questo script fa debug per capire perché ICP fallisce sempre.
Testa su un singolo campione e stampa tutte le informazioni.
"""

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from plyfile import PlyData
from scipy.spatial.transform import Rotation as Rot
import cv2

# =============================================================================
# FUNZIONI DI UTILITY
# =============================================================================

def depth_to_pointcloud(depth, fx, fy, cx, cy, mask=None):
    """Convert depth image to 3D point cloud."""
    H, W = depth.shape
    
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    points = np.stack([X, Y, Z], axis=-1)
    
    if mask is not None:
        valid = mask & (depth > 0.01) & (depth < 10.0)
    else:
        valid = (depth > 0.01) & (depth < 10.0)
    
    return points[valid], valid.sum()


def transform_points(points, R, t):
    """Transform 3D points."""
    return points @ R.T + t


def quaternion_to_matrix(quat):
    """Convert quaternion [w,x,y,z] to rotation matrix."""
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    return Rot.from_quat(quat_xyzw).as_matrix()


def icp_simple(source_points, target_points, max_iterations=30, tolerance=1e-6):
    """
    Simple point-to-point ICP using SVD.
    Fallback puro Python senza dipendenze esterne.
    """
    from scipy.spatial import cKDTree
    
    src = source_points.copy()
    transform = np.eye(4)
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        # Find nearest neighbors
        tree = cKDTree(target_points)
        distances, indices = tree.query(src, k=1)
        
        # Filter outliers (distance > 5cm)
        inlier_mask = distances < 0.05
        if inlier_mask.sum() < 100:
            print(f"   ICP iter {iteration}: too few inliers ({inlier_mask.sum()})")
            return transform, float('inf'), False
        
        src_inliers = src[inlier_mask]
        tgt_inliers = target_points[indices[inlier_mask]]
        
        # Compute centroids
        centroid_src = np.mean(src_inliers, axis=0)
        centroid_tgt = np.mean(tgt_inliers, axis=0)
        
        # Center points
        src_centered = src_inliers - centroid_src
        tgt_centered = tgt_inliers - centroid_tgt
        
        # SVD for optimal rotation
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation
        t = centroid_tgt - R @ centroid_src
        
        # Apply
        src = (R @ src.T).T + t
        
        # Update transform
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        transform = T_iter @ transform
        
        # Check convergence
        mean_error = np.mean(distances[inlier_mask])
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    return transform, mean_error, True


# =============================================================================
# DEBUG MAIN
# =============================================================================

def main():
    print("="*70)
    print("ICP DEBUG - Testing on single sample")
    print("="*70)
    
    # Config
    dataset_root = '/Users/emanuelerosapepe/Desktop/test_YOLO/Linemod_preprocessed'
    split_val = 'data/autosplit_val_ALL.txt'
    model_dir = 'checkpoints/'
    base_weights = 'DenseFusion_Masked_DualAtt_NetVar.pth'
    
    # Camera intrinsics
    fx, fy = 572.4114, 573.57043
    cx, cy = 325.2611, 242.04899
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset
    print("\n📦 Loading dataset...")
    from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset
    
    val_ds = LineModPoseDataset(split_val, dataset_root, mode='val')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f"   Loaded {len(val_ds)} samples")
    
    # Load 3D model (just one for testing)
    print("\n📐 Loading 3D model...")
    models_3d = {}
    models_dir = os.path.join(dataset_root, 'models')
    
    DIAMETERS = {
        1: 102.09, 2: 247.50, 4: 172.49, 5: 201.40, 6: 154.54,
        8: 261.47, 9: 108.99, 10: 164.62, 11: 175.88, 12: 145.54,
        13: 278.07, 14: 282.60, 15: 212.35
    }
    
    for obj_id in DIAMETERS.keys():
        path = os.path.join(models_dir, f"obj_{obj_id:02d}.ply")
        if os.path.exists(path):
            ply = PlyData.read(path)
            vertex = ply['vertex']
            pts = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)
            models_3d[obj_id] = pts / 1000.0  # mm -> m
    
    print(f"   Loaded {len(models_3d)} models")
    
    # Load pose model
    print("\n🧠 Loading pose model...")
    from models.DFMasked_DualAtt_NetVar import DenseFusion_Masked_DualAtt_NetVar
    
    model = DenseFusion_Masked_DualAtt_NetVar(pretrained=False, temperature=2.0).to(device)
    weights_path = os.path.join(model_dir, base_weights)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("   ✓ Model loaded")
    
    # Test on first few samples
    print("\n" + "="*70)
    print("TESTING ICP ON SAMPLES")
    print("="*70)
    
    n_test = 5
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= n_test:
                break
            
            print(f"\n--- Sample {idx+1} ---")
            
            # Get data
            rgb = batch['image'].to(device)
            depth_tensor = batch['depth'].to(device)
            mask_tensor = batch['mask'].to(device)
            bb_info = batch['bbox_norm'].to(device)
            cam_params = batch['cam_params'].to(device)
            gt_quat = batch['quaternion'][0].numpy()
            gt_trans = batch['translation'][0].numpy()
            obj_id = int(batch['class_id'][0])
            
            print(f"   Object ID: {obj_id}")
            
            if obj_id not in models_3d:
                print(f"   ⚠️  Model not found for obj {obj_id}")
                continue
            
            # Get prediction
            pred_quat, pred_trans = model(rgb, depth_tensor, bb_info, cam_params, mask_tensor)
            pred_q = pred_quat[0].cpu().numpy()
            pred_t = pred_trans[0].cpu().numpy()
            
            print(f"   Pred translation: [{pred_t[0]:.4f}, {pred_t[1]:.4f}, {pred_t[2]:.4f}]")
            print(f"   GT translation:   [{gt_trans[0]:.4f}, {gt_trans[1]:.4f}, {gt_trans[2]:.4f}]")
            
            # Extract depth
            depth = depth_tensor[0].squeeze().cpu().numpy()
            print(f"\n   Depth stats:")
            print(f"      Shape: {depth.shape}")
            print(f"      Min: {depth.min():.4f}")
            print(f"      Max: {depth.max():.4f}")
            print(f"      Mean: {depth.mean():.4f}")
            print(f"      Non-zero: {(depth > 0).sum()}")
            
            # Determine depth format and convert to meters
            if depth.max() <= 1.0:
                print(f"      Format: Normalized [0-1]")
                # Need to figure out the actual scale
                # LineMOD depth is typically in mm, max ~2000mm
                depth_m = depth * 2.0  # Assume normalized to 2m max
            elif depth.max() > 100:
                print(f"      Format: Millimeters")
                depth_m = depth / 1000.0
            else:
                print(f"      Format: Meters (assumed)")
                depth_m = depth
            
            print(f"      Depth (m) range: [{depth_m.min():.4f}, {depth_m.max():.4f}]")
            
            # Extract mask
            mask = mask_tensor[0].squeeze().cpu().numpy()
            mask_bool = mask > 0.5
            print(f"\n   Mask stats:")
            print(f"      Shape: {mask.shape}")
            print(f"      True pixels: {mask_bool.sum()}")
            print(f"      Mask coverage: {mask_bool.sum() / mask.size * 100:.2f}%")
            
            # Convert to point cloud
            target_points, n_target = depth_to_pointcloud(
                depth_m, fx, fy, cx, cy, mask_bool
            )
            print(f"\n   Target point cloud:")
            print(f"      Points: {n_target}")
            if n_target > 0:
                print(f"      X range: [{target_points[:,0].min():.4f}, {target_points[:,0].max():.4f}]")
                print(f"      Y range: [{target_points[:,1].min():.4f}, {target_points[:,1].max():.4f}]")
                print(f"      Z range: [{target_points[:,2].min():.4f}, {target_points[:,2].max():.4f}]")
            
            # Transform model points
            model_pts = models_3d[obj_id]
            # Subsample
            if len(model_pts) > 1000:
                indices = np.random.choice(len(model_pts), 1000, replace=False)
                model_pts = model_pts[indices]
            
            pred_R = quaternion_to_matrix(pred_q)
            source_points = transform_points(model_pts, pred_R, pred_t)
            
            print(f"\n   Source point cloud (transformed model):")
            print(f"      Points: {len(source_points)}")
            print(f"      X range: [{source_points[:,0].min():.4f}, {source_points[:,0].max():.4f}]")
            print(f"      Y range: [{source_points[:,1].min():.4f}, {source_points[:,1].max():.4f}]")
            print(f"      Z range: [{source_points[:,2].min():.4f}, {source_points[:,2].max():.4f}]")
            
            # Check overlap
            if n_target < 100:
                print(f"\n   ❌ Not enough target points for ICP!")
                continue
            
            # Compute initial distance
            from scipy.spatial import cKDTree
            tree = cKDTree(target_points)
            dists, _ = tree.query(source_points, k=1)
            print(f"\n   Initial alignment:")
            print(f"      Mean distance: {np.mean(dists)*100:.2f} cm")
            print(f"      Median distance: {np.median(dists)*100:.2f} cm")
            print(f"      Points within 2cm: {(dists < 0.02).sum()}/{len(dists)}")
            print(f"      Points within 5cm: {(dists < 0.05).sum()}/{len(dists)}")
            
            # Pre-align with centroids
            centroid_src = np.mean(source_points, axis=0)
            centroid_tgt = np.mean(target_points, axis=0)
            offset = centroid_tgt - centroid_src
            print(f"\n   Centroid offset: [{offset[0]*100:.2f}, {offset[1]*100:.2f}, {offset[2]*100:.2f}] cm")
            
            source_adjusted = source_points + offset
            
            # Check after centroid alignment
            dists2, _ = tree.query(source_adjusted, k=1)
            print(f"\n   After centroid alignment:")
            print(f"      Mean distance: {np.mean(dists2)*100:.2f} cm")
            print(f"      Points within 2cm: {(dists2 < 0.02).sum()}/{len(dists2)}")
            print(f"      Points within 5cm: {(dists2 < 0.05).sum()}/{len(dists2)}")
            
            # Try ICP
            print(f"\n   Running ICP...")
            transform, error, success = icp_simple(
                source_adjusted, target_points, max_iterations=30
            )
            
            print(f"   ICP result:")
            print(f"      Success: {success}")
            print(f"      Final error: {error*100:.2f} cm")
            
            if success:
                # Apply transform and check
                src_final = (transform[:3,:3] @ source_adjusted.T).T + transform[:3,3]
                dists3, _ = tree.query(src_final, k=1)
                print(f"      Mean distance after ICP: {np.mean(dists3)*100:.2f} cm")
                print(f"      Points within 2cm: {(dists3 < 0.02).sum()}/{len(dists3)}")
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()