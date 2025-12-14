import numpy as np
import cv2

class PoseEvaluator:
    def __init__(self):
        """
        camera_intrinsics: Matrice 3x3 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """


    def calculate_add_metric(self, pred_R, pred_t, gt_R, gt_t, model_3d_points):
        """
        Calcola la metrica ADD (Average Distance of Model Points).
        
        pred_R: Matrice rotazione 3x3 predetta (dalla PoseNet)
        pred_t: Vettore traslazione predetto (dalla funzione sopra)
        gt_R, gt_t: Ground Truth
        model_3d_points: Nuvola di punti (Nx3) dal file .ply dell'oggetto
        """
        # 1. Applica posa VERA ai punti del modello
        # P_true = (R_gt * P) + t_gt
        target_points = np.dot(model_3d_points, gt_R.T) + gt_t
        
        # 2. Applica posa PREDETTA ai punti del modello
        # P_pred = (R_pred * P) + t_pred
        estimated_points = np.dot(model_3d_points, pred_R.T) + pred_t
        
        # 3. Calcola la distanza Euclidea media tra i punti corrispondenti
        # ADD = avg( || P_true - P_pred || )
        distances = np.linalg.norm(target_points - estimated_points, axis=1)
        add_score = np.mean(distances)
        
        return add_score

    def is_pose_correct(self, add_score, diameter_threshold=0.1):
        """
        Considera la posa corretta se l'errore ADD è < 10% del diametro dell'oggetto.
        """
        return add_score < diameter_threshold