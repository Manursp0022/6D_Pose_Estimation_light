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

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("⚠️  Open3D not found. Install with: pip install open3d")

class ICPRefiner:
    """
    ICP-based pose refinement for 6D pose estimation.
    
    Usage:
        refiner = ICPRefiner(models_3d, camera_intrinsics)
        refined_R, refined_t = refiner.refine(
            pred_R, pred_t, depth, mask, obj_id
        )
    """
    
    def __init__(
        self,
        models_3d,
        cam_K,
        max_correspondence_distance=0.02,  # 2cm
        max_iterations=50,
        n_model_points=1000,
        method='open3d'  # 'open3d' or 'cv2'
    ):
        """
        Args:
            models_3d: dict {obj_id: [N, 3] model points in meters}
            cam_K: [3, 3] camera intrinsic matrix
            max_correspondence_distance: max distance for ICP matching
            max_iterations: ICP iterations
            n_model_points: number of model points to use (subsample for speed)
            method: 'open3d' or 'cv2'
        """
        self.models_3d = models_3d
        self.cam_K = cam_K
        self.fx = cam_K[0, 0]
        self.fy = cam_K[1, 1]
        self.cx = cam_K[0, 2]
        self.cy = cam_K[1, 2]
        
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iterations = max_iterations
        self.n_model_points = n_model_points
        self.method = method
        
        # Subsample model points for speed
        self.subsampled_models = {}
        for obj_id, points in models_3d.items():
            if len(points) > n_model_points:
                indices = np.random.choice(len(points), n_model_points, replace=False)
                self.subsampled_models[obj_id] = points[indices]
            else:
                self.subsampled_models[obj_id] = points
    
    def refine(self, pred_R, pred_t, depth, mask, obj_id, 
               min_points=500, return_debug=False):
        """
        Refine a single pose prediction using ICP.
        
        Args:
            pred_R: [3, 3] predicted rotation matrix
            pred_t: [3] predicted translation vector
            depth: [H, W] depth image in meters
            mask: [H, W] object mask (boolean)
            obj_id: int, object ID
            min_points: minimum points required for ICP
            return_debug: return debug info
            
        Returns:
            refined_R: [3, 3] refined rotation
            refined_t: [3] refined translation
            success: bool, whether refinement was successful
        """
        # Get model points
        if obj_id not in self.subsampled_models:
            if return_debug:
                return pred_R, pred_t, False, {'error': 'Unknown object ID'}
            return pred_R, pred_t, False
        
        model_points = self.subsampled_models[obj_id]
        
        # Transform model points with initial pose
        source_points = transform_points(model_points, pred_R, pred_t)
        
        # Extract target point cloud from depth
        target_points = depth_to_pointcloud(
            depth, self.fx, self.fy, self.cx, self.cy, mask
        )
        
        # Check if we have enough points
        if len(target_points) < min_points or len(source_points) < min_points:
            if return_debug:
                return pred_R, pred_t, False, {
                    'error': f'Not enough points: source={len(source_points)}, target={len(target_points)}'
                }
            return pred_R, pred_t, False
        
        # Adjust initial translation using centroids (from CosyPose)
        centroid_src = np.mean(source_points, axis=0)
        centroid_tgt = np.mean(target_points, axis=0)
        centroid_offset = centroid_tgt - centroid_src
        
        # Apply centroid adjustment
        source_points_adjusted = source_points + centroid_offset
        pred_t_adjusted = pred_t + centroid_offset
        
        try:
            if self.method == 'open3d' and HAS_OPEN3D:
                # Run Open3D ICP
                transform, fitness, rmse = icp_open3d(
                    source_points_adjusted, target_points,
                    max_correspondence_distance=self.max_correspondence_distance,
                    max_iterations=self.max_iterations
                )
                
                # Check if ICP succeeded
                if fitness < 0.3:  # Less than 30% overlap
                    if return_debug:
                        return pred_R, pred_t, False, {
                            'error': f'Low fitness: {fitness:.3f}',
                            'fitness': fitness, 'rmse': rmse
                        }
                    return pred_R, pred_t, False
                
            else:
                # Fall back to CV2 ICP
                transform, residual, retval = icp_cv2(
                    source_points_adjusted, target_points,
                    tolerance=0.05, max_iterations=self.max_iterations
                )
                
                if retval == -1 or residual > 0.05:
                    if return_debug:
                        return pred_R, pred_t, False, {
                            'error': f'CV2 ICP failed: retval={retval}, residual={residual}'
                        }
                    return pred_R, pred_t, False
                
                fitness = 1.0  # CV2 doesn't return fitness
                rmse = residual
            
            # Extract refined pose
            # transform maps source_adjusted -> target
            # So: refined_R = transform[:3,:3] @ pred_R
            #     refined_t = transform[:3,:3] @ pred_t_adjusted + transform[:3,3]
            
            delta_R = transform[:3, :3]
            delta_t = transform[:3, 3]
            
            refined_R = delta_R @ pred_R
            refined_t = delta_R @ pred_t_adjusted + delta_t
            
            if return_debug:
                return refined_R, refined_t, True, {
                    'fitness': fitness,
                    'rmse': rmse,
                    'centroid_offset': centroid_offset,
                    'n_source': len(source_points),
                    'n_target': len(target_points)
                }
            
            return refined_R, refined_t, True
            
        except Exception as e:
            if return_debug:
                return pred_R, pred_t, False, {'error': str(e)}
            return pred_R, pred_t, False
    
    def refine_batch(self, pred_Rs, pred_ts, depths, masks, obj_ids):
        """
        Refine a batch of pose predictions.
        
        Args:
            pred_Rs: list of [3, 3] rotation matrices
            pred_ts: list of [3] translation vectors
            depths: list of [H, W] depth images
            masks: list of [H, W] masks
            obj_ids: list of object IDs
            
        Returns:
            refined_Rs: list of refined rotations
            refined_ts: list of refined translations
            successes: list of success flags
        """
        refined_Rs = []
        refined_ts = []
        successes = []
        
        for pred_R, pred_t, depth, mask, obj_id in zip(
            pred_Rs, pred_ts, depths, masks, obj_ids
        ):
            ref_R, ref_t, success = self.refine(pred_R, pred_t, depth, mask, obj_id)
            refined_Rs.append(ref_R)
            refined_ts.append(ref_t)
            successes.append(success)
        
        return refined_Rs, refined_ts, successes