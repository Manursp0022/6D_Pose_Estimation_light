"""
ICP Pose Refiner - Simplified Version for LineMOD

This module refines 6D pose estimates using ICP (Iterative Closest Point).
Adapted from CosyPose's ICP refiner but simplified to work without rendering.

Key idea:
1. Take initial pose prediction from your model
2. Transform 3D model points with predicted pose
3. Extract point cloud from depth image
4. Run ICP to align model points to observed depth
5. Return refined pose

Requirements:
    pip install open3d

Author: Adapted from CosyPose (Labbé et al., 2020)
"""

import numpy as np
import torch
import cv2
from scipy import ndimage
from utils.ICP_utils.ICP_functions import get_normal, depth_to_pointcloud, transform_points, icp_open3d, icp_cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot


def depth_to_pointcloud_from_crop(depth, bbox_norm, cam_params_norm, 
                                   crop_size=224, orig_size=(480, 640),
                                   pad_factor=1.2, mask=None):
    """
    Convert CROPPED depth image back to 3D point cloud in camera coordinates.
    
    This function reverses the crop_square_resize operation to get correct 3D points.
    
    Args:
        depth: [crop_size, crop_size] cropped depth in meters
        bbox_norm: [cx, cy, w, h] normalized bbox (center format, normalized to [0,1])
        cam_params_norm: [fx, fy, cx, cy] normalized camera params
        crop_size: size of cropped image (224)
        orig_size: (H, W) of original image (480, 640)
        pad_factor: padding factor used in crop (1.2)
        mask: optional [crop_size, crop_size] mask
        
    Returns:
        points: [N, 3] point cloud in camera coordinates (meters)
    """
    H_orig, W_orig = orig_size
    
    # Denormalize camera parameters
    fx = cam_params_norm[0] * W_orig
    fy = cam_params_norm[1] * H_orig
    cx_cam = cam_params_norm[2] * W_orig  # Principal point
    cy_cam = cam_params_norm[3] * H_orig
    
    # Denormalize bbox (it's in center format: [cx, cy, w, h])
    bbox_cx = bbox_norm[0] * W_orig  # Center x of bbox in original image
    bbox_cy = bbox_norm[1] * H_orig  # Center y of bbox in original image
    bbox_w = bbox_norm[2] * W_orig   # Width of bbox
    bbox_h = bbox_norm[3] * H_orig   # Height of bbox
    
    # Calculate the square crop parameters (same as crop_square_resize)
    side = max(bbox_w, bbox_h) * pad_factor
    
    # Top-left corner of the square crop in original image
    crop_x1 = bbox_cx - side / 2
    crop_y1 = bbox_cy - side / 2
    
    # Scale factor: how many original pixels per crop pixel
    scale = side / crop_size
    
    # Create pixel grid in crop coordinates [0, crop_size)
    u_crop = np.arange(crop_size)
    v_crop = np.arange(crop_size)
    u_crop, v_crop = np.meshgrid(u_crop, v_crop)
    
    # Map crop pixels to original image coordinates
    # u_orig = crop_x1 + u_crop * scale + scale/2  (center of each pixel)
    u_orig = crop_x1 + (u_crop + 0.5) * scale
    v_orig = crop_y1 + (v_crop + 0.5) * scale
    
    # Get depth values
    Z = depth
    
    # Unproject to 3D using pinhole camera model
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    X = (u_orig - cx_cam) * Z / fx
    Y = (v_orig - cy_cam) * Z / fy
    
    # Stack to [H, W, 3]
    points = np.stack([X, Y, Z], axis=-1)
    
    # Filter valid points
    if mask is not None:
        valid = mask & (Z > 0.1) & (Z < 5.0) & np.isfinite(Z)
    else:
        valid = (Z > 0.1) & (Z < 5.0) & np.isfinite(Z)
    
    points = points[valid]
    
    # Remove any remaining NaN/Inf
    valid_points = np.isfinite(points).all(axis=1)
    points = points[valid_points]
    
    return points


def transform_points(points, R, t):
    """Transform 3D points with rotation and translation."""
    points = np.asarray(points, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).flatten()
    return points @ R.T + t


def icp_point_to_point(source, target, max_iterations=50, tolerance=1e-6, max_dist=0.05):
    """
    Point-to-point ICP using SVD.
    
    Args:
        source: [N, 3] source points (model transformed with predicted pose)
        target: [M, 3] target points (from depth)
        max_iterations: maximum iterations
        tolerance: convergence tolerance
        max_dist: max distance for inlier matching
        
    Returns:
        R: [3, 3] rotation correction
        t: [3] translation correction
        success: bool
        final_error: float
    """
    src = np.asarray(source, dtype=np.float64).copy()
    tgt = np.asarray(target, dtype=np.float64)
    
    if len(src) < 50 or len(tgt) < 50:
        return np.eye(3), np.zeros(3), False, float('inf')
    
    tree = cKDTree(tgt)
    
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        distances, indices = tree.query(src, k=1)
        
        # Inlier selection with adaptive threshold
        inlier_mask = distances < max_dist
        n_inliers = inlier_mask.sum()
        
        if n_inliers < 50:
            # Try with larger distance
            inlier_mask = distances < max_dist * 2
            n_inliers = inlier_mask.sum()
            if n_inliers < 50:
                # Still not enough - try even larger
                inlier_mask = distances < max_dist * 4
                n_inliers = inlier_mask.sum()
                if n_inliers < 50:
                    return R_total, t_total, False, float('inf')
        
        src_inliers = src[inlier_mask]
        tgt_inliers = tgt[indices[inlier_mask]]
        
        # Compute centroids
        centroid_src = np.mean(src_inliers, axis=0)
        centroid_tgt = np.mean(tgt_inliers, axis=0)
        
        # Center points
        src_centered = src_inliers - centroid_src
        tgt_centered = tgt_inliers - centroid_tgt
        
        # SVD for optimal rotation
        H = src_centered.T @ tgt_centered
        
        if not np.isfinite(H).all():
            return R_total, t_total, False, float('inf')
        
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation
        t = centroid_tgt - R @ centroid_src
        
        # Apply transformation
        src = (R @ src.T).T + t
        
        # Accumulate
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        # Check convergence
        mean_error = np.mean(distances[inlier_mask])
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    # Final error
    distances_final, _ = tree.query(src, k=1)
    final_error = np.mean(distances_final)
    
    return R_total, t_total, True, final_error


class ICPRefiner:
    """
    ICP-based pose refinement for 6D pose estimation.
    
    FINAL VERSION - correctly handles cropped images.
    """
    
    def __init__(
        self,
        models_3d,
        cam_K,
        max_correspondence_distance=0.05,
        max_iterations=50,
        n_model_points=1000,
        orig_img_size=(480, 640),
        crop_size=224,
        pad_factor=1.2,
        **kwargs
    ):
        """
        Args:
            models_3d: dict {obj_id: [N, 3] model points in meters}
            cam_K: [3, 3] camera intrinsic matrix (for original image)
            max_correspondence_distance: max distance for ICP matching (meters)
            max_iterations: ICP iterations
            n_model_points: number of model points to use
            orig_img_size: (H, W) of original image
            crop_size: size of cropped image
            pad_factor: padding factor used in crop_square_resize
        """
        self.models_3d = models_3d
        self.cam_K = np.array(cam_K, dtype=np.float64)
        
        self.orig_img_size = orig_img_size
        self.crop_size = crop_size
        self.pad_factor = pad_factor
        
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iterations = max_iterations
        self.n_model_points = n_model_points
        
        # Subsample model points for speed
        self.subsampled_models = {}
        for obj_id, points in models_3d.items():
            pts = np.asarray(points, dtype=np.float64)
            if len(pts) > n_model_points:
                np.random.seed(42)  # Reproducibility
                indices = np.random.choice(len(pts), n_model_points, replace=False)
                self.subsampled_models[obj_id] = pts[indices]
            else:
                self.subsampled_models[obj_id] = pts
        
        print(f"   ICP Refiner initialized:")
        print(f"      - Max correspondence dist: {max_correspondence_distance*100:.1f} cm")
        print(f"      - Max iterations: {max_iterations}")
        print(f"      - Model points: {n_model_points}")
    
    def refine(self, pred_R, pred_t, depth, mask, obj_id,
               bbox_norm=None, cam_params_norm=None,
               min_points=200, return_debug=False):
        """
        Refine a single pose prediction using ICP.
        
        Args:
            pred_R: [3, 3] predicted rotation matrix
            pred_t: [3] predicted translation vector (meters)
            depth: [H, W] cropped depth image (meters)
            mask: [H, W] object mask (boolean or float)
            obj_id: int, object ID
            bbox_norm: [4] normalized bbox [cx, cy, w, h] used for cropping
            cam_params_norm: [4] normalized camera params [fx, fy, cx, cy]
            min_points: minimum points required for ICP
            return_debug: return debug info
            
        Returns:
            refined_R: [3, 3] refined rotation
            refined_t: [3] refined translation
            success: bool
            debug_info: dict (if return_debug=True)
        """
        debug_info = {}
        
        # Convert to proper types
        pred_R = np.asarray(pred_R, dtype=np.float64)
        pred_t = np.asarray(pred_t, dtype=np.float64).flatten()
        depth = np.asarray(depth, dtype=np.float64)
        mask = np.asarray(mask) > 0.5  # Ensure boolean
        
        # Check model exists
        if obj_id not in self.subsampled_models:
            if return_debug:
                return pred_R, pred_t, False, {'error': f'Unknown object ID: {obj_id}'}
            return pred_R, pred_t, False
        
        model_points = self.subsampled_models[obj_id]
        
        # Transform model points with predicted pose
        source_points = transform_points(model_points, pred_R, pred_t)
        
        # Extract target point cloud from depth
        if bbox_norm is not None and cam_params_norm is not None:
            bbox_norm = np.asarray(bbox_norm, dtype=np.float64)
            cam_params_norm = np.asarray(cam_params_norm, dtype=np.float64)
            
            target_points = depth_to_pointcloud_from_crop(
                depth,
                bbox_norm=bbox_norm,
                cam_params_norm=cam_params_norm,
                crop_size=self.crop_size,
                orig_size=self.orig_img_size,
                pad_factor=self.pad_factor,
                mask=mask
            )
        else:
            # Fallback - should not happen if evaluator passes the params
            if return_debug:
                return pred_R, pred_t, False, {'error': 'Missing bbox_norm or cam_params_norm'}
            return pred_R, pred_t, False
        
        debug_info['n_source'] = len(source_points)
        debug_info['n_target'] = len(target_points)
        
        # Check minimum points
        if len(target_points) < min_points:
            if return_debug:
                return pred_R, pred_t, False, {
                    'error': f'Not enough target points: {len(target_points)}',
                    **debug_info
                }
            return pred_R, pred_t, False
        
        # Compute centroids
        centroid_source = np.mean(source_points, axis=0)
        centroid_target = np.mean(target_points, axis=0)
        centroid_offset = centroid_target - centroid_source
        
        debug_info['centroid_offset_cm'] = np.linalg.norm(centroid_offset) * 100
        
        # Pre-align using centroids
        source_aligned = source_points + centroid_offset
        
        # Run ICP
        R_icp, t_icp, success, final_error = icp_point_to_point(
            source_aligned,
            target_points,
            max_iterations=self.max_iterations,
            tolerance=1e-6,
            max_dist=self.max_correspondence_distance
        )
        
        debug_info['icp_success'] = success
        debug_info['icp_error_cm'] = final_error * 100
        
        if not success:
            if return_debug:
                return pred_R, pred_t, False, {'error': 'ICP failed', **debug_info}
            return pred_R, pred_t, False
        
        # Compute refined pose
        # The ICP transform maps source_aligned to target
        # source_aligned = source + centroid_offset = R_pred @ model + t_pred + centroid_offset
        # After ICP: R_icp @ source_aligned + t_icp ≈ target
        # So: R_refined = R_icp @ R_pred
        #     t_refined = R_icp @ (t_pred + centroid_offset) + t_icp
        
        refined_R = R_icp @ pred_R
        refined_t = R_icp @ (pred_t + centroid_offset) + t_icp
        
        # Ensure rotation is orthogonal
        U, _, Vt = np.linalg.svd(refined_R)
        refined_R = U @ Vt
        if np.linalg.det(refined_R) < 0:
            U[:, -1] *= -1
            refined_R = U @ Vt
        
        debug_info['success'] = True
        debug_info['translation_change_cm'] = np.linalg.norm(refined_t - pred_t) * 100
        
        if return_debug:
            return refined_R, refined_t, True, debug_info
        
        return refined_R, refined_t, True
    
    def refine_batch(self, pred_Rs, pred_ts, depths, masks, obj_ids,
                     bbox_norms=None, cam_params_norms=None, min_points=200):
        """Refine a batch of pose predictions."""
        refined_Rs = []
        refined_ts = []
        successes = []
        
        for i, (pred_R, pred_t, depth, mask, obj_id) in enumerate(
            zip(pred_Rs, pred_ts, depths, masks, obj_ids)
        ):
            bbox_norm = bbox_norms[i] if bbox_norms is not None else None
            cam_params_norm = cam_params_norms[i] if cam_params_norms is not None else None
            
            ref_R, ref_t, success = self.refine(
                pred_R, pred_t, depth, mask, obj_id,
                bbox_norm=bbox_norm,
                cam_params_norm=cam_params_norm,
                min_points=min_points
            )
            refined_Rs.append(ref_R)
            refined_ts.append(ref_t)
            successes.append(success)
        
        return refined_Rs, refined_ts, successes


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def quaternion_to_matrix_np(quat):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    return Rot.from_quat(quat_xyzw).as_matrix()


def matrix_to_quaternion_np(R):
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    r = Rot.from_matrix(R)
    quat_xyzw = r.as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
