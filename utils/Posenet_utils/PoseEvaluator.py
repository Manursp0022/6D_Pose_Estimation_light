import numpy as np
import cv2
from scipy.spatial import KDTree

class PoseEvaluator:
    def __init__(self, camera_intrinsics):
        """
        camera_intrinsics: Matrice 3x3 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        self.K = camera_intrinsics
        self.fx = camera_intrinsics[0, 0]
        self.fy = camera_intrinsics[1, 1]
        self.cx = camera_intrinsics[0, 2]
        self.cy = camera_intrinsics[1, 2]

        self.symmetric_obj_ids = [10, 11]

    def estimate_translation(self, bbox_2d, real_obj_diameter_mm):
        x, y, w, h = bbox_2d
        
        # Compute 2D center (u, v)
        u = x + w / 2
        v = y + h / 2
        
        # depth estimation (Tz)
        # We use the maximum size of the box (diagonal or side) for robustness.
        # We assume that the apparent size in pixels is proportional to the distance.
        # Note: This is an estimate (PnP approximation).
        # Tz = (focal length * actual size) / apparent size
        box_size_px = max(w, h)
        tz = (self.fx * real_obj_diameter_mm) / box_size_px
        
        # Back-projection per Tx e Ty
        tx = (u - self.cx) * tz / self.fx
        ty = (v - self.cy) * tz / self.fy
        
        return np.array([tx, ty, tz])

    def calculate_metric(self,pred_R, pred_t, gt_R, gt_t, model_3d_points, obj_id):
        target_points = np.dot(model_3d_points, gt_R.T) + gt_t #model 3d points id the ply model that is multiplicated for gorunf truth rotation so we create the "Target"

        estimated_points = np.dot(model_3d_points, pred_R.T) + pred_t

        if obj_id in self.symmetric_obj_ids : 
            return self._calculate_add_s(target_points, estimated_points)
        else:
            return self._calculate_add(target_points, estimated_points)
    def calculate_separated_metrics(self, pred_R, pred_t, gt_R, gt_t, model_3d_points, obj_id):
        """
        Calcola separatamente:
        1. Errore Traslazione (distanza euclidea centri)
        2. Errore Rotazione 'Fisico' (ADD applicato solo alla rotazione, ignorando T)
        3. Errore Rotazione 'Angolare' (Gradi)
        """
        # 1. TRANSLATION ERROR (Euclidean Distance in meters)
        trans_error = np.linalg.norm(gt_t - pred_t)

        # 2. ROTATION ERROR (Physical displacement in meters)
        # Applichiamo SOLO la rotazione ai punti (T=0) per isolare l'errore rotazionale
        target_points_rot_only = np.dot(model_3d_points, gt_R.T)
        est_points_rot_only = np.dot(model_3d_points, pred_R.T)

        if obj_id in self.symmetric_obj_ids:
            rot_error_m = self._calculate_add_s(target_points_rot_only, est_points_rot_only)
        else:
            rot_error_m = self._calculate_add(target_points_rot_only, est_points_rot_only)

        # 3. ROTATION ERROR (Angular in Degrees)
        # Calcolo R_diff = R_gt * R_pred^T
        # Trace method: theta = arccos( (tr(R) - 1) / 2 )
        R_diff = np.dot(gt_R, pred_R.T)
        trace = np.trace(R_diff)
        
        # Clamp per evitare errori numerici fuori da [-1, 1]
        trace = np.clip((trace - 1) / 2, -1.0, 1.0)
        rot_error_deg = np.degrees(np.arccos(trace))

        return trans_error, rot_error_m, rot_error_deg

    def calculate_separated_metrics(self, pred_R, pred_t, gt_R, gt_t, model_3d_points, obj_id):
            """
            Calcola separatamente:
            1. Errore Traslazione Totale (Euclideo)
            2. Errore Traslazione per asse (X, Y, Z)  <-- NUOVO
            3. Errore Rotazione 'Fisico'
            4. Errore Rotazione 'Angolare'
            """
            # 1. TRANSLATION ERROR (Euclidean Distance in meters)
            trans_error = np.linalg.norm(gt_t - pred_t)
            
            # --- NUOVO: Calcolo errore per singolo asse ---
            # gt_t e pred_t sono vettori [x, y, z]
            diff = np.abs(gt_t - pred_t)
            tx_err = diff[0]
            ty_err = diff[1]
            tz_err = diff[2]
            # ----------------------------------------------

            # 2. ROTATION ERROR (Physical displacement in meters)
            target_points_rot_only = np.dot(model_3d_points, gt_R.T)
            est_points_rot_only = np.dot(model_3d_points, pred_R.T)

            if obj_id in self.symmetric_obj_ids:
                rot_error_m = self._calculate_add_s(target_points_rot_only, est_points_rot_only)
            else:
                rot_error_m = self._calculate_add(target_points_rot_only, est_points_rot_only)

            # 3. ROTATION ERROR (Angular in Degrees)
            R_diff = np.dot(gt_R, pred_R.T)
            trace = np.trace(R_diff)
            trace = np.clip((trace - 1) / 2, -1.0, 1.0)
            rot_error_deg = np.degrees(np.arccos(trace))

            # Restituiamo anche i singoli errori su assi
            return trans_error, rot_error_m, rot_error_deg, tx_err, ty_err, tz_err
    
    def _calculate_add(self, target_points, estimated_points):
        """
        ADD Classic (Average distance of model points) for asymmetrical objects.
        Formula: avg( || (Rx + t) - (R'x + t') || )
        """
        distances = np.linalg.norm(target_points - estimated_points, axis=1)
        return np.mean(distances)

    def _calculate_add_s(self, target_points, estimated_points):
        """
        ADD-S (Symmetric) for symmetric objects (Eggbox, Glue).
        Formula: avg( min || (Rx + t) - (R'x + t') || )
        Uses KDTree to efficiently find the nearest neighbor.
        """
        tree = KDTree(estimated_points)
        
        # For each true point (target), we find the distance to the nearest estimated point.
        # k=1 means “find only the nearest neighbor.”
        distances, _ = tree.query(target_points, k=1)
        return np.mean(distances)

    def is_pose_correct(self, add_score, diameter_threshold=0.1):
        """
        Consider the placement correct if the ADD error is < 10% of the object diameter.
        """
        return add_score < diameter_threshold