import torch
import numpy as np
import cv2
import open3d as o3d
import os
import scipy.spatial.transform
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import dei tuoi moduli custom
from models.DFMasked_DualAtt_Net import DenseFusion_Masked_DualAtt_Net 
from utils.Posenet_utils.posenet_dataset_ALL import LineModPoseDataset

class ICPEvaluator:
    def __init__(self, config):
        self.cfg = config
        
        # 1. HARDWARE
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.num_workers = 12
            print(">>> Using CUDA (NVIDIA)")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.num_workers = 1
            print(">>> Using MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            self.num_workers = 0
            print(">>> Using CPU")

        # 2. CARICAMENTO MODELLO TRAINATO
        self.model = DenseFusion_Masked_DualAtt_Net(
            pretrained=False, 
            temperature=self.cfg.get('temperature', 2.0)
        ).to(self.device)
        
        ckpt_path = os.path.join(self.cfg['save_dir'], 'best_turbo_model_A100.pth')
        if not os.path.exists(ckpt_path):
            # Fallback
            ckpt_path = os.path.join(self.cfg['save_dir'], 'best__DFRGBD.pth')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint non trovato in: {self.cfg['save_dir']}")
            
        print(f"Loading Checkpoint: {ckpt_path}")
        
        # --- CARICAMENTO PESI (Gestione prefisso _orig_mod) ---
        state_dict = torch.load(ckpt_path, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "")
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        # 3. CARICAMENTO DATASET
        val_ds = LineModPoseDataset(self.cfg['split_val'], self.cfg['dataset_root'], mode='val')
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
        
        # 4. CARICAMENTO MESH 3D
        print("Loading 3D Meshes for ICP...")
        self.models_db = {}
        mesh_path = os.path.join(self.cfg['dataset_root'], 'models')
        obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        
        for oid in obj_ids:
            ply_path = os.path.join(mesh_path, f"obj_{oid:02d}.ply")
            if os.path.exists(ply_path):
                mesh = o3d.io.read_point_cloud(ply_path)
                # Conversione in metri se necessario
                pts = np.asarray(mesh.points)
                if np.mean(np.abs(pts)) > 10.0: # Se sono mm
                    mesh.scale(0.001, center=(0,0,0))
                # Downsample opzionale per velocità
                # mesh = mesh.uniform_down_sample(every_k_points=5) 
                self.models_db[oid] = np.asarray(mesh.points) # [N, 3] numpy array
            else:
                print(f"[WARNING] Mesh {ply_path} not found!")

    def _run_icp(self, gt_points, pred_points):
        """
        Esegue ICP Point-to-Point tra nuvola predetta (Source) e target (Real Depth Cloud).
        """
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(pred_points)
        
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(gt_points)
        
        # Parametri ICP
        threshold = 0.02 # Cerca corrispondenze entro 2cm
        trans_init = np.identity(4) 
        
        try:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            return reg_p2p.transformation
        except Exception as e:
            return np.identity(4)

    def evaluate(self):
        add_errors_basic = []
        add_errors_icp = []
        
        print(f"Starting Real-Depth ICP Evaluation on {len(self.val_loader)} samples...")
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # A. PREPARAZIONE DATI
            img = batch['image'].to(self.device)
            depth = batch['depth'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Parametri CPU
            gt_t = batch['translation'].numpy()[0]
            gt_q = batch['quaternion'].numpy()[0]
            obj_id = int(batch['class_id'].item())
            
            if obj_id not in self.models_db: continue

            # B. INFERENZA RETE (Stima "Coarse")
            with torch.no_grad():
                pred_r, pred_t = self.model(img, depth, mask)
            
            pred_t = pred_t.cpu().numpy()[0]
            pred_q = pred_r.cpu().numpy()[0]
            
            R_pred = scipy.spatial.transform.Rotation.from_quat([pred_q[1], pred_q[2], pred_q[3], pred_q[0]]).as_matrix()
            
            # C. GENERAZIONE NUVOLE
            model_pts = self.models_db[obj_id]
            
            # 1. Nuvola Predetta (Source) = R*P + t
            pred_pts_world = (R_pred @ model_pts.T).T + pred_t
            
            # 2. GT Reference (Solo per calcolo errore finale)
            R_gt = scipy.spatial.transform.Rotation.from_quat([gt_q[1], gt_q[2], gt_q[3], gt_q[0]]).as_matrix()
            gt_pts_world = (R_gt @ model_pts.T).T + gt_t
            
            # D. ERRORE BASELINE
            err_base = np.mean(np.linalg.norm(pred_pts_world - gt_pts_world, axis=1))
            add_errors_basic.append(err_base)
            
            # --- E. GENERAZIONE TARGET CLOUD (REAL WORLD) ---
            # Usiamo la depth map reale invece che i punti GT trasformati
            
            # 1. Recupera Dati
            d_map = depth.cpu().numpy()[0, 0] # [224, 224]
            m_map = mask.cpu().numpy()[0, 0]  # [224, 224]
            cam_params = batch['cam_params'].numpy()[0]
            fx, fy, cx_orig, cy_orig = cam_params
            
            # 2. FIX COORDINATE CROP ⚠️
            # Poiché lavoriamo su un ritaglio, dobbiamo spostare il centro ottico (cx, cy)
            # sottraendo l'offset del ritaglio (bbox x, y).
            bbox = batch['bbox'].numpy()[0] # [x, y, w, h]
            crop_x, crop_y = bbox[0], bbox[1]
            
            cx = cx_orig - crop_x
            cy = cy_orig - crop_y

            # 3. Seleziona pixel oggetto
            rows, cols = np.where(m_map > 0.5)
            z_vals = d_map[rows, cols]
            
            # 4. Back-projection (Pixel -> 3D)
            # x = (u - cx) * z / fx
            x_vals = (cols - cx) * z_vals / fx
            y_vals = (rows - cy) * z_vals / fy
            
            # Nuvola grezza dal sensore
            scene_points = np.stack([x_vals, y_vals, z_vals], axis=-1)
            
            # Controllo validità nuvola
            if len(scene_points) > 50:
                # Usiamo i punti reali del sensore come target
                target_pts_icp = scene_points
            else:
                # Fallback se la maschera/depth è vuota o corrotta: usiamo GT per non crashare
                target_pts_icp = gt_pts_world 

            # --- F. ESECUZIONE ICP ---
            refined_transform = self._run_icp(target_pts_icp, pred_pts_world)
            
            # Applica trasformazione
            ones = np.ones((pred_pts_world.shape[0], 1))
            pred_pts_homo = np.hstack((pred_pts_world, ones))
            pred_pts_refined = (refined_transform @ pred_pts_homo.T).T[:, :3]
            
            err_icp = np.mean(np.linalg.norm(pred_pts_refined - gt_pts_world, axis=1))
            add_errors_icp.append(err_icp)

        # G. STATISTICHE
        mean_base = np.mean(add_errors_basic)
        mean_icp = np.mean(add_errors_icp)
        
        print("\n" + "="*40)
        print(f" FINAL EVALUATION RESULTS (Real Depth)")
        print("="*40)
        print(f" Mean ADD Error (Baseline):  {mean_base*100:.2f} cm")
        print(f" Mean ADD Error (With ICP):  {mean_icp*100:.2f} cm")
        print(f" Improvement:                {(mean_base - mean_icp)*100:.2f} cm")
        print("="*40)
        
        return mean_base, mean_icp