import numpy as np
import torch
import cv2
from scipy import ndimage
# =============================================================================
# UTILITY FUNCTIONS (from CosyPose)
# =============================================================================

def get_normal(depth_refine, fx, fy, cx, cy, bbox=np.array([0]), refine=True):
    """
    Fast normal computation from depth map.
    From: https://github.com/kirumang/Pix2Pose
    """
    res_y = depth_refine.shape[0]
    res_x = depth_refine.shape[1]
    
    if refine:
        depth_refine = np.nan_to_num(depth_refine)
        mask = np.zeros_like(depth_refine).astype(np.uint8)
        mask[depth_refine == 0] = 1
        depth_refine = depth_refine.astype(np.float32)
        depth_refine = cv2.inpaint(depth_refine, mask, 2, cv2.INPAINT_NS)
        depth_refine = depth_refine.astype(np.float64)
        depth_refine = ndimage.gaussian_filter(depth_refine, 2)

    uv_table = np.zeros((res_y, res_x, 2), dtype=np.int16)
    column = np.arange(0, res_y)
    uv_table[:, :, 1] = np.arange(0, res_x) - cx
    uv_table[:, :, 0] = column[:, np.newaxis] - cy

    constant_x = 1 / fx
    constant_y = 1 / fy

    if bbox.shape[0] == 4:
        uv_table = uv_table[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        v_x = np.zeros((bbox[2]-bbox[0], bbox[3]-bbox[1], 3))
        v_y = np.zeros((bbox[2]-bbox[0], bbox[3]-bbox[1], 3))
        normals = np.zeros((bbox[2]-bbox[0], bbox[3]-bbox[1], 3))
        depth_refine = depth_refine[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    else:
        v_x = np.zeros((res_y, res_x, 3))
        v_y = np.zeros((res_y, res_x, 3))
        normals = np.zeros((res_y, res_x, 3))

    uv_table_sign = np.copy(uv_table)
    uv_table = np.abs(np.copy(uv_table))

    dig = np.gradient(depth_refine, 2, edge_order=2)
    v_y[:, :, 0] = uv_table_sign[:, :, 1] * constant_x * dig[0]
    v_y[:, :, 1] = depth_refine * constant_y + (uv_table_sign[:, :, 0] * constant_y) * dig[0]
    v_y[:, :, 2] = dig[0]

    v_x[:, :, 0] = depth_refine * constant_x + uv_table_sign[:, :, 1] * constant_x * dig[1]
    v_x[:, :, 1] = uv_table_sign[:, :, 0] * constant_y * dig[1]
    v_x[:, :, 2] = dig[1]

    cross = np.cross(v_x.reshape(-1, 3), v_y.reshape(-1, 3))
    norm = np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)
    norm[norm == 0] = 1
    cross = cross / norm
    
    if bbox.shape[0] == 4:
        cross = cross.reshape((bbox[2]-bbox[0], bbox[3]-bbox[1], 3))
    else:
        cross = cross.reshape(res_y, res_x, 3)
    
    cross = np.nan_to_num(cross)
    return cross


def depth_to_pointcloud(depth, fx, fy, cx, cy, mask=None):
    """
    Convert depth image to 3D point cloud.
    
    Args:
        depth: [H, W] depth image in meters
        fx, fy, cx, cy: camera intrinsics
        mask: [H, W] optional mask for valid points
        
    Returns:
        points: [N, 3] point cloud
    """
    H, W = depth.shape
    
    # Create pixel grid
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    # Backproject to 3D
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack to [H, W, 3]
    points = np.stack([X, Y, Z], axis=-1)
    
    # Apply mask if provided
    if mask is not None:
        valid = mask & (depth > 0.01) & (depth < 10.0)  # Valid depth range
    else:
        valid = (depth > 0.01) & (depth < 10.0)
    
    points = points[valid]
    
    return points


def transform_points(points, R, t):
    """
    Transform 3D points with rotation and translation.
    
    Args:
        points: [N, 3]
        R: [3, 3] rotation matrix
        t: [3] translation vector
        
    Returns:
        transformed: [N, 3]
    """
    return points @ R.T + t


# =============================================================================
# ICP REFINEMENT FUNCTIONS
# =============================================================================

def icp_open3d(source_points, target_points, init_transform=None, 
               max_correspondence_distance=0.02, max_iterations=50):
    """
    Run ICP using Open3D.
    
    Args:
        source_points: [N, 3] model points (transformed with initial pose)
        target_points: [M, 3] observed points from depth
        init_transform: [4, 4] initial transformation (usually identity after pre-transform)
        max_correspondence_distance: max distance for point matching
        max_iterations: ICP iterations
        
    Returns:
        transform: [4, 4] refined transformation
        fitness: float, overlap ratio
        rmse: float, RMSE of aligned points
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D required for ICP. Install with: pip install open3d")
    
    # Create Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    
    # Estimate normals (helps with ICP)
    source_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    
    # Initial transform
    if init_transform is None:
        init_transform = np.eye(4)
    
    # Run ICP
    # Option 1: Point-to-Point ICP
    # result = o3d.pipelines.registration.registration_icp(
    #     source_pcd, target_pcd, max_correspondence_distance, init_transform,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    # )
    
    # Option 2: Point-to-Plane ICP (usually better)
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    return result.transformation, result.fitness, result.inlier_rmse


def icp_cv2(source_points, target_points, source_normals=None, target_normals=None,
            tolerance=0.05, max_iterations=100, num_levels=4):
    """
    Run ICP using OpenCV's ppf_match_3d_ICP (like CosyPose).
    
    Args:
        source_points: [N, 3] model points
        target_points: [M, 3] observed points
        source_normals: [N, 3] optional normals
        target_normals: [M, 3] optional normals
        tolerance: convergence tolerance
        max_iterations: max ICP iterations
        num_levels: pyramid levels
        
    Returns:
        transform: [4, 4] refined transformation
        residual: float, alignment residual
        retval: int, success flag
    """
    # Prepare points with normals (6D: xyz + normal)
    if source_normals is None:
        source_normals = np.zeros_like(source_points)
    if target_normals is None:
        target_normals = np.zeros_like(target_points)
    
    source_6d = np.hstack([source_points, source_normals]).astype(np.float32)
    target_6d = np.hstack([target_points, target_normals]).astype(np.float32)
    
    # Run ICP
    icp = cv2.ppf_match_3d_ICP(max_iterations, tolerence=tolerance, numLevels=num_levels)
    retval, residual, pose = icp.registerModelToScene(source_6d.reshape(-1, 6), 
                                                       target_6d.reshape(-1, 6))
    
    return pose, residual, retval